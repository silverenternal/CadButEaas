//! 自适应容差系统
//!
//! ## 设计哲学
//!
//! 传统的 CAD 系统使用硬编码阈值（如 `0.5mm`），但这在不同场景下表现不佳：
//! - 建筑图纸（米单位，坐标值 0-100）：0.5mm 容差太小
//! - 机械图纸（毫米单位，坐标值 0-10000）：0.5mm 容差太大
//! - 地图数据（经纬度，坐标值 -180 到 180）：0.5mm 容差无意义
//!
//! 本模块实现**动态自适应容差**，基于：
//! 1. 图纸单位（从 $INSUNITS 解析）
//! 2. 场景特征尺度（从坐标范围计算）
//! 3. 用户操作精度（从交互行为推断）
//!
//! ## 使用示例
//!
//! ```rust
//! use common_types::adaptive_tolerance::{AdaptiveTolerance, PrecisionLevel};
//! use common_types::scene::LengthUnit;
//!
//! let tolerance = AdaptiveTolerance::new(
//!     LengthUnit::Mm,
//!     1000.0,  // 场景尺度：1000mm
//!     PrecisionLevel::Normal,
//! );
//!
//! // 动态计算的吸附容差
//! let snap_tol = tolerance.snap_tolerance();  // 约 0.5mm
//!
//! // 动态 Bulge 阈值（基于弦长）
//! let bulge_threshold = tolerance.bulge_threshold(100.0);  // 弦长 100mm 时的 bulge 阈值
//! ```

use crate::scene::LengthUnit;
use serde::{Deserialize, Serialize};

/// 精度级别（从用户行为推断）
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PrecisionLevel {
    /// 粗加工：容差放大 10 倍（快速但不精确）
    Rough,
    /// 正常：标准容差（默认）
    Normal,
    /// 精加工：容差缩小 10 倍（精确但慢）
    Fine,
    /// 超精密：容差缩小 100 倍（极高精度）
    Ultra,
}

impl Default for PrecisionLevel {
    fn default() -> Self {
        Self::Normal
    }
}

/// 自适应容差计算器
///
/// ## 核心公式
///
/// ### 1. 吸附容差
/// ```text
/// snap_tolerance = base_tolerance × scale_factor × precision_factor
///
/// 其中：
/// - base_tolerance: 基于单位的基础容差（米：0.0005, 毫米：0.5, 英寸：0.02）
/// - scale_factor: 基于场景尺度的调整因子（大场景放大，小场景缩小）
/// - precision_factor: 基于用户精度的调整因子（Rough: 10x, Normal: 1x, Fine: 0.1x, Ultra: 0.01x）
/// ```
///
/// ### 2. Bulge 阈值
/// ```text
/// bulge_threshold = 2 × max_sagitta / chord_length
///
/// 其中：
/// - max_sagitta: 允许的最大拱高（snap_tolerance × 0.1）
/// - chord_length: 弦长
///
/// 物理意义：当 bulge 产生的拱高小于容差的 1/10 时，简化为直线
/// ```
///
/// ### 3. 交点容差
/// ```text
/// intersection_tolerance = scene_scale × 1e-6
///
/// 物理意义：相对容差，避免大坐标场景下的浮点误差
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveTolerance {
    /// 基础单位（从图纸解析）
    pub base_unit: LengthUnit,
    /// 场景特征尺度（从坐标范围计算）
    pub scene_scale: f64,
    /// 用户操作精度（从交互推断）
    pub operation_precision: PrecisionLevel,
}

impl AdaptiveTolerance {
    /// 创建自适应容差计算器
    ///
    /// ## 参数
    /// - `base_unit`: 图纸单位（从 DXF $INSUNITS 解析）
    /// - `scene_scale`: 场景特征尺度（通常是坐标范围的对角线长度）
    /// - `operation_precision`: 用户操作精度级别
    ///
    /// ## 示例
    /// ```rust,no_run
    /// use common_types::adaptive_tolerance::{AdaptiveTolerance, PrecisionLevel};
    /// use common_types::scene::LengthUnit;
    ///
    /// let tol = AdaptiveTolerance::new(
    ///     LengthUnit::Mm,
    ///     1000.0,  // 1 米 x 1 米的场景
    ///     PrecisionLevel::Normal,
    /// );
    /// ```
    pub fn new(
        base_unit: LengthUnit,
        scene_scale: f64,
        operation_precision: PrecisionLevel,
    ) -> Self {
        Self {
            base_unit,
            scene_scale,
            operation_precision,
        }
    }

    /// 从实体列表自动计算场景尺度
    ///
    /// ## 使用示例
    /// ```rust,no_run
    /// use common_types::adaptive_tolerance::{AdaptiveTolerance, PrecisionLevel};
    /// use common_types::scene::LengthUnit;
    /// use common_types::geometry::RawEntity;
    ///
    /// let entities: Vec<RawEntity> = vec![];
    /// let tolerance = AdaptiveTolerance::from_entities(
    ///     LengthUnit::Mm,
    ///     &entities,
    ///     PrecisionLevel::Normal,
    /// );
    /// ```
    pub fn from_entities(
        base_unit: LengthUnit,
        entities: &[crate::geometry::RawEntity],
        operation_precision: PrecisionLevel,
    ) -> Self {
        let scene_scale = Self::compute_scene_scale(entities);
        Self::new(base_unit, scene_scale, operation_precision)
    }

    /// 计算场景特征尺度（坐标范围的对角线长度）
    fn compute_scene_scale(entities: &[crate::geometry::RawEntity]) -> f64 {
        if entities.is_empty() {
            return 1000.0;  // 默认 1 米
        }

        let mut min_x = f64::INFINITY;
        let mut max_x = f64::NEG_INFINITY;
        let mut min_y = f64::INFINITY;
        let mut max_y = f64::NEG_INFINITY;

        for entity in entities {
            match entity {
                crate::geometry::RawEntity::Line { start, end, .. } => {
                    min_x = min_x.min(start[0]).min(end[0]);
                    max_x = max_x.max(start[0]).max(end[0]);
                    min_y = min_y.min(start[1]).min(end[1]);
                    max_y = max_y.max(start[1]).max(end[1]);
                }
                crate::geometry::RawEntity::Polyline { points, .. } => {
                    for pt in points {
                        min_x = min_x.min(pt[0]);
                        max_x = max_x.max(pt[0]);
                        min_y = min_y.min(pt[1]);
                        max_y = max_y.max(pt[1]);
                    }
                }
                crate::geometry::RawEntity::Arc { center, radius, .. }
                | crate::geometry::RawEntity::Circle { center, radius, .. } => {
                    min_x = min_x.min(center[0] - radius);
                    max_x = max_x.max(center[0] + radius);
                    min_y = min_y.min(center[1] - radius);
                    max_y = max_y.max(center[1] + radius);
                }
                crate::geometry::RawEntity::Text { position, .. } => {
                    min_x = min_x.min(position[0]);
                    max_x = max_x.max(position[0]);
                    min_y = min_y.min(position[1]);
                    max_y = max_y.max(position[1]);
                }
                _ => {}
            }
        }

        let width = max_x - min_x;
        let height = max_y - min_y;
        (width * width + height * height).sqrt()
    }

    /// 获取基于单位的基础容差（毫米）
    fn base_tolerance(&self) -> f64 {
        match self.base_unit {
            LengthUnit::Mm => 0.5,
            LengthUnit::Cm => 0.05,        // 0.5mm in cm
            LengthUnit::M => 0.0005,       // 0.5mm in meters
            LengthUnit::Inch => 0.02,      // 0.5mm in inches
            LengthUnit::Foot => 0.0016,    // 0.5mm in feet
            LengthUnit::Yard => 0.00055,   // 0.5mm in yards
            LengthUnit::Mile => 0.00000031, // 0.5mm in miles
            LengthUnit::Micron => 500.0,   // 0.5mm in microns
            LengthUnit::Kilometer => 0.0000005, // 0.5mm in kilometers
            LengthUnit::Point => 1.42,     // 0.5mm in points
            LengthUnit::Pica => 0.118,     // 0.5mm in picas
            LengthUnit::Unspecified => 0.5, // 假设毫米
        }
    }

    /// 计算场景尺度调整因子
    ///
    /// ## 启发式规则
    /// - 场景尺度 > 10000：大场景，容差放大 10 倍
    /// - 场景尺度 < 100：小场景，容差缩小到 1/10
    /// - 其他：不调整
    fn scale_factor(&self) -> f64 {
        if self.scene_scale > 10000.0 {
            10.0  // 大场景（如总图），容差放大
        } else if self.scene_scale < 100.0 {
            0.1   // 小场景（如零件图），容差缩小
        } else {
            1.0   // 正常场景
        }
    }

    /// 计算用户精度调整因子
    fn precision_factor(&self) -> f64 {
        match self.operation_precision {
            PrecisionLevel::Rough => 10.0,
            PrecisionLevel::Normal => 1.0,
            PrecisionLevel::Fine => 0.1,
            PrecisionLevel::Ultra => 0.01,
        }
    }

    // ========================================================================
    // 公共 API：动态容差计算
    // ========================================================================

    /// 计算自适应吸附容差（单位：毫米）
    ///
    /// ## 用途
    /// - 端点吸附：合并距离小于此容差的端点
    /// - 点捕捉：鼠标点击时捕捉到最近的端点/交点
    ///
    /// ## 返回
    /// 动态计算的容差值（毫米）
    ///
    /// ## 示例
    /// ```rust,no_run
    /// use common_types::adaptive_tolerance::{AdaptiveTolerance, PrecisionLevel};
    /// use common_types::scene::LengthUnit;
    ///
    /// let tol = AdaptiveTolerance::new(LengthUnit::Mm, 1000.0, PrecisionLevel::Normal);
    /// let snap_tol = tol.snap_tolerance();  // 约 0.5mm
    /// ```
    pub fn snap_tolerance(&self) -> f64 {
        self.base_tolerance() * self.scale_factor() * self.precision_factor()
    }

    /// 计算动态 Bulge 阈值
    ///
    /// ## 用途
    /// 判断 Bulge 是否需要离散化为圆弧，还是可以直接简化为直线
    ///
    /// ## 参数
    /// - `chord_length`: 弦长（起点到终点的距离）
    ///
    /// ## 核心思想
    /// Bulge 产生的拱高（sagitta）公式：`h = L × bulge / 2`
    ///
    /// 我们希望 `h < snap_tolerance × 0.1`，所以：
    /// ```text
    /// bulge_threshold = 2 × max_sagitta / chord_length
    ///                 = 2 × (snap_tolerance × 0.1) / chord_length
    /// ```
    ///
    /// ## 物理意义
    /// 当 bulge 产生的拱高小于容差的 1/10 时，视为直线（人眼无法分辨）
    ///
    /// ## 示例
    /// ```rust,no_run
    /// use common_types::adaptive_tolerance::{AdaptiveTolerance, PrecisionLevel};
    /// use common_types::scene::LengthUnit;
    ///
    /// let tol = AdaptiveTolerance::new(LengthUnit::Mm, 1000.0, PrecisionLevel::Normal);
    /// let threshold = tol.bulge_threshold(100.0);  // 弦长 100mm 时的阈值
    /// ```
    pub fn bulge_threshold(&self, chord_length: f64) -> f64 {
        let tolerance = self.snap_tolerance();
        let max_sagitta = tolerance * 0.1;  // 拱高小于容差的 1/10

        // bulge = 2 * sagitta / L
        // 防止除以零，chord_length 至少为 tolerance
        2.0 * max_sagitta / chord_length.max(tolerance)
    }

    /// 计算动态交点容差
    ///
    /// ## 用途
    /// - 判断两条线段是否相交
    /// - 交点去重
    ///
    /// ## 核心思想
    /// 使用相对容差：`scene_scale × 1e-6`
    ///
    /// 对于大坐标场景（如地图，坐标值 1e6），容差为 1.0
    /// 对于小坐标场景（如零件，坐标值 100），容差为 0.0001
    ///
    /// ## 返回
    /// 动态计算的交点容差值
    pub fn intersection_tolerance(&self) -> f64 {
        self.scene_scale * 1e-6
    }

    /// 计算动态弦高误差容差（用于 NURBS 离散化）
    ///
    /// ## 用途
    /// 控制 NURBS 曲线离散化的精度
    ///
    /// ## 核心思想
    /// 基于屏幕空间误差而非世界空间误差
    ///
    /// ## 参数
    /// - `zoom`: 当前缩放级别（可选）
    ///
    /// ## 返回
    /// 动态计算的弦高误差容差
    pub fn chord_tolerance(&self, zoom: Option<f64>) -> f64 {
        let base = self.snap_tolerance();

        // 如果提供了缩放级别，基于屏幕空间调整
        if let Some(z) = zoom {
            // 缩放越大（放大），容差越小（更精确）
            base / z.max(0.1)
        } else {
            base
        }
    }

    /// 计算动态最小边长（用于去噪）
    ///
    /// ## 用途
    /// 移除长度小于此值的短边/碎线
    ///
    /// ## 返回
    /// 动态计算的最小边长（snap_tolerance 的 2 倍）
    pub fn min_edge_length(&self) -> f64 {
        self.snap_tolerance() * 2.0
    }

    /// 计算动态角度容差（用于共线检测）
    ///
    /// ## 用途
    /// 判断两条线段是否共线
    ///
    /// ## 返回
    /// 动态计算的角度容差（弧度）
    pub fn angle_tolerance(&self) -> f64 {
        // 默认 5 度，根据精度级别调整
        let base_degrees: f64 = match self.operation_precision {
            PrecisionLevel::Rough => 10.0,
            PrecisionLevel::Normal => 5.0,
            PrecisionLevel::Fine => 1.0,
            PrecisionLevel::Ultra => 0.1,
        };
        base_degrees.to_radians()
    }

    // ========================================================================
    // 工具方法
    // ========================================================================

    /// 获取当前容差配置的摘要信息
    pub fn summary(&self) -> String {
        format!(
            "AdaptiveTolerance {{\n\
             \t单位：{:?}\n\
             \t场景尺度：{:.2}\n\
             \t精度级别：{:?}\n\
             \t吸附容差：{:.6}\n\
             \t交点容差：{:.6}\n\
             \t最小边长：{:.6}\n\
             }}",
            self.base_unit,
            self.scene_scale,
            self.operation_precision,
            self.snap_tolerance(),
            self.intersection_tolerance(),
            self.min_edge_length(),
        )
    }
}

impl Default for AdaptiveTolerance {
    /// 默认配置：毫米单位，1 米场景，正常精度
    fn default() -> Self {
        Self::new(
            LengthUnit::Mm,
            1000.0,
            PrecisionLevel::Normal,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_base_tolerance() {
        let tol = AdaptiveTolerance::new(LengthUnit::Mm, 1000.0, PrecisionLevel::Normal);
        assert!((tol.base_tolerance() - 0.5).abs() < 1e-10);

        let tol_m = AdaptiveTolerance::new(LengthUnit::M, 1000.0, PrecisionLevel::Normal);
        assert!((tol_m.base_tolerance() - 0.0005).abs() < 1e-10);
    }

    #[test]
    fn test_snap_tolerance() {
        let tol = AdaptiveTolerance::new(LengthUnit::Mm, 1000.0, PrecisionLevel::Normal);
        assert!((tol.snap_tolerance() - 0.5).abs() < 1e-10);

        // 大场景放大
        let tol_large = AdaptiveTolerance::new(LengthUnit::Mm, 20000.0, PrecisionLevel::Normal);
        assert!((tol_large.snap_tolerance() - 5.0).abs() < 1e-10);

        // 小场景缩小
        let tol_small = AdaptiveTolerance::new(LengthUnit::Mm, 50.0, PrecisionLevel::Normal);
        assert!((tol_small.snap_tolerance() - 0.05).abs() < 1e-10);
    }

    #[test]
    fn test_precision_factor() {
        let tol_rough = AdaptiveTolerance::new(LengthUnit::Mm, 1000.0, PrecisionLevel::Rough);
        assert!((tol_rough.snap_tolerance() - 5.0).abs() < 1e-10);

        let tol_fine = AdaptiveTolerance::new(LengthUnit::Mm, 1000.0, PrecisionLevel::Fine);
        assert!((tol_fine.snap_tolerance() - 0.05).abs() < 1e-10);
    }

    #[test]
    fn test_bulge_threshold() {
        let tol = AdaptiveTolerance::new(LengthUnit::Mm, 1000.0, PrecisionLevel::Normal);

        // 弦长 100mm 时，bulge 阈值约 0.001
        let threshold_100 = tol.bulge_threshold(100.0);
        assert!(threshold_100 < 0.01);

        // 弦长越短，bulge 阈值越大（更容易简化）
        let threshold_10 = tol.bulge_threshold(10.0);
        assert!(threshold_10 > threshold_100);
    }

    #[test]
    fn test_intersection_tolerance() {
        let tol = AdaptiveTolerance::new(LengthUnit::Mm, 1000.0, PrecisionLevel::Normal);
        assert!((tol.intersection_tolerance() - 0.001).abs() < 1e-10);

        // 大坐标场景
        let tol_large = AdaptiveTolerance::new(LengthUnit::Mm, 1000000.0, PrecisionLevel::Normal);
        assert!((tol_large.intersection_tolerance() - 1.0).abs() < 1e-10);
    }
}
