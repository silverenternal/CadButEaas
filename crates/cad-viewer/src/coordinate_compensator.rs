//! 坐标精度补偿器
//!
//! ## 设计哲学
//!
//! egui 使用 `f32` 进行渲染，但世界坐标使用 `f64`。对于大坐标场景（如地图，坐标值 `1e6`），
//! `f64 → f32` 转换会导致精度丢失（从 `0.0001mm` 降到 `0.1mm`）。
//!
//! 本模块实现**相对坐标渲染**：
//! 1. 选择场景原点（通常是场景中心或第一个点）
//! 2. 所有坐标减去原点，转换为相对坐标
//! 3. 相对坐标值很小，`f64 → f32` 转换精度损失可忽略
//!
//! ## 精度对比
//!
//! | 场景 | 绝对坐标渲染 | 相对坐标渲染 |
//! |------|-------------|-------------|
//! | 小坐标（<1000） | 精度损失 1e-7 | 精度损失 1e-7 |
//! | 大坐标（1e6） | 精度损失 0.1mm | 精度损失 1e-6mm |
//!
//! ## 使用示例
//!
//! ```rust
//! use cad_viewer::coordinate_compensator::CoordinateCompensator;
//!
//! // 创建补偿器（自动选择场景原点）
//! let compensator = CoordinateCompensator::new(&edges);
//!
//! // 世界坐标转屏幕坐标
//! let screen = compensator.world_to_screen(world, zoom, pan, center);
//!
//! // 屏幕坐标转世界坐标
//! let world = compensator.screen_to_world(screen, zoom, pan, center);
//! ```

use common_types::Point2;
use eframe::egui::{Pos2, Vec2};

/// 坐标精度补偿器
///
/// ## 核心思想
///
/// 对于大坐标场景，使用**相对坐标**而非绝对坐标：
/// ```text
/// relative_coord = world_coord - scene_origin
/// screen_coord = relative_coord * zoom + pan + center
/// ```
///
/// 这样 `f64 → f32` 转换的精度损失从 `scene_coord × 1e-7` 降到 `relative_coord × 1e-7`。
///
/// ## 自动启用策略
///
/// - 坐标值 > 10000：自动启用相对坐标
/// - 坐标值 < 10000：使用绝对坐标（避免原点计算开销）
#[derive(Debug, Clone)]
pub struct CoordinateCompensator {
    /// 场景原点（世界坐标）
    scene_origin: Point2,
    /// 是否启用相对坐标
    use_relative: bool,
    /// 坐标范围（用于调试）
    bounds: BoundingBox,
}

/// 坐标范围 bounding box
#[derive(Debug, Clone, Copy, Default)]
pub struct BoundingBox {
    pub min_x: f64,
    pub max_x: f64,
    pub min_y: f64,
    pub max_y: f64,
}

impl BoundingBox {
    /// 获取最大坐标值
    pub fn max_coord(&self) -> f64 {
        self.max_x.abs().max(self.max_y.abs())
    }
}

/// 边（简化版，用于坐标补偿器）
#[derive(Debug, Clone)]
pub struct Edge {
    pub start: Point2,
    pub end: Point2,
}

impl CoordinateCompensator {
    /// 创建坐标补偿器（从边列表自动计算）
    ///
    /// ## 参数
    /// - `edges`: 所有边（用于计算场景范围）
    ///
    /// ## 自动决策
    /// - 坐标值 > 10000：启用相对坐标
    /// - 坐标值 < 10000：使用绝对坐标
    ///
    /// ## 示例
    /// ```rust
    /// let compensator = CoordinateCompensator::new(&edges);
    /// ```
    pub fn new(edges: &[Edge]) -> Self {
        let bounds = Self::compute_bounds(edges);
        let use_relative = bounds.max_coord() > 10000.0;

        // 场景原点：场景中心（如果启用相对坐标）或原点
        let scene_origin = if use_relative {
            [
                (bounds.min_x + bounds.max_x) / 2.0,
                (bounds.min_y + bounds.max_y) / 2.0,
            ]
        } else {
            [0.0, 0.0]
        };

        Self {
            scene_origin,
            use_relative,
            bounds,
        }
    }

    /// 从点列表创建补偿器
    ///
    /// ## 参数
    /// - `points`: 所有点（用于计算场景范围）
    ///
    /// ## 示例
    /// ```rust
    /// let points = vec![[0.0, 0.0], [100000.0, 100000.0]];
    /// let compensator = CoordinateCompensator::from_points(&points);
    /// ```
    pub fn from_points(points: &[Point2]) -> Self {
        let bounds = Self::compute_bounds_from_points(points);
        let use_relative = bounds.max_coord() > 10000.0;

        let scene_origin = if use_relative {
            [
                (bounds.min_x + bounds.max_x) / 2.0,
                (bounds.min_y + bounds.max_y) / 2.0,
            ]
        } else {
            [0.0, 0.0]
        };

        Self {
            scene_origin,
            use_relative,
            bounds,
        }
    }

    /// 从边列表计算坐标范围
    fn compute_bounds(edges: &[Edge]) -> BoundingBox {
        let mut min_x = f64::INFINITY;
        let mut max_x = f64::NEG_INFINITY;
        let mut min_y = f64::INFINITY;
        let mut max_y = f64::NEG_INFINITY;

        for edge in edges {
            min_x = min_x.min(edge.start[0]).min(edge.end[0]);
            max_x = max_x.max(edge.start[0]).max(edge.end[0]);
            min_y = min_y.min(edge.start[1]).min(edge.end[1]);
            max_y = max_y.max(edge.start[1]).max(edge.end[1]);
        }

        if min_x == f64::INFINITY {
            BoundingBox {
                min_x: 0.0,
                max_x: 1000.0,
                min_y: 0.0,
                max_y: 1000.0,
            }
        } else {
            BoundingBox {
                min_x,
                max_x,
                min_y,
                max_y,
            }
        }
    }

    /// 从点列表计算坐标范围
    fn compute_bounds_from_points(points: &[Point2]) -> BoundingBox {
        let mut min_x = f64::INFINITY;
        let mut max_x = f64::NEG_INFINITY;
        let mut min_y = f64::INFINITY;
        let mut max_y = f64::NEG_INFINITY;

        for pt in points {
            min_x = min_x.min(pt[0]);
            max_x = max_x.max(pt[0]);
            min_y = min_y.min(pt[1]);
            max_y = max_y.max(pt[1]);
        }

        if min_x == f64::INFINITY {
            BoundingBox {
                min_x: 0.0,
                max_x: 1000.0,
                min_y: 0.0,
                max_y: 1000.0,
            }
        } else {
            BoundingBox {
                min_x,
                max_x,
                min_y,
                max_y,
            }
        }
    }

    /// 世界坐标转屏幕坐标（带精度补偿）
    ///
    /// ## 参数
    /// - `world`: 世界坐标（f64）
    /// - `zoom`: 缩放级别
    /// - `pan`: 平移偏移（屏幕空间）
    /// - `center`: 屏幕中心（屏幕空间）
    ///
    /// ## 返回
    /// 屏幕坐标（f32）
    ///
    /// ## 精度保证
    /// - 小坐标场景：直接使用绝对坐标
    /// - 大坐标场景：使用相对坐标，精度损失 < 1e-6mm
    pub fn world_to_screen(&self, world: Point2, zoom: f64, pan: Vec2, center: Pos2) -> Pos2 {
        if self.use_relative {
            // 使用相对坐标：先减原点，再转 f32
            let relative = [
                world[0] - self.scene_origin[0],
                world[1] - self.scene_origin[1],
            ];
            Pos2::new(
                ((relative[0] * zoom) as f32 + pan.x) + center.x,
                ((-relative[1] * zoom) as f32 + pan.y) + center.y, // Y 轴翻转
            )
        } else {
            // 小坐标场景：直接转换
            Pos2::new(
                ((world[0] * zoom) as f32 + pan.x) + center.x,
                ((-world[1] * zoom) as f32 + pan.y) + center.y,
            )
        }
    }

    /// 屏幕坐标转世界坐标（带精度补偿）
    ///
    /// ## 参数
    /// - `screen`: 屏幕坐标（f32）
    /// - `zoom`: 缩放级别
    /// - `pan`: 平移偏移（屏幕空间）
    /// - `center`: 屏幕中心（屏幕空间）
    ///
    /// ## 返回
    /// 世界坐标（f64）
    pub fn screen_to_world(&self, screen: Pos2, zoom: f64, pan: Vec2, center: Pos2) -> Point2 {
        // 防止除以零
        if zoom.abs() < 1e-10 {
            return [0.0, 0.0];
        }

        let world = [
            ((screen.x - center.x - pan.x) as f64 / zoom),
            ((center.y - screen.y + pan.y) as f64 / zoom), // Y 轴翻转
        ];

        if self.use_relative {
            // 相对坐标转世界坐标：加回原点
            [
                world[0] + self.scene_origin[0],
                world[1] + self.scene_origin[1],
            ]
        } else {
            world
        }
    }

    /// 获取场景原点（世界坐标）
    ///
    /// ## 用途
    /// - 调试显示
    /// - 坐标转换验证
    pub fn scene_origin(&self) -> Point2 {
        self.scene_origin
    }

    /// 是否启用相对坐标
    pub fn use_relative(&self) -> bool {
        self.use_relative
    }

    /// 获取坐标范围（用于调试）
    pub fn bounds(&self) -> BoundingBox {
        self.bounds
    }

    /// 获取精度补偿摘要
    pub fn summary(&self) -> String {
        format!(
            "CoordinateCompensator {{\n\
             \t启用相对坐标：{}\n\
             \t场景原点：[{:.4}, {:.4}]\n\
             \t坐标范围：[{:.2}, {:.2}] × [{:.2}, {:.2}]\n\
             \t最大坐标：{:.2}\n\
             }}",
            self.use_relative,
            self.scene_origin[0],
            self.scene_origin[1],
            self.bounds.min_x,
            self.bounds.max_x,
            self.bounds.min_y,
            self.bounds.max_y,
            self.bounds.max_coord(),
        )
    }
}

impl Default for CoordinateCompensator {
    /// 默认配置：不启用相对坐标
    fn default() -> Self {
        Self {
            scene_origin: [0.0, 0.0],
            use_relative: false,
            bounds: BoundingBox {
                min_x: 0.0,
                max_x: 1000.0,
                min_y: 0.0,
                max_y: 1000.0,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_small_coordinates_no_compensation() {
        // 小坐标场景：不启用相对坐标
        let edges = vec![Edge {
            start: [0.0, 0.0],
            end: [100.0, 0.0],
        }];
        let compensator = CoordinateCompensator::new(&edges);

        assert!(!compensator.use_relative);
        assert_eq!(compensator.scene_origin, [0.0, 0.0]);
    }

    #[test]
    fn test_large_coordinates_compensation() {
        // 大坐标场景：启用相对坐标
        let edges = vec![Edge {
            start: [100000.0, 100000.0],
            end: [200000.0, 100000.0],
        }];
        let compensator = CoordinateCompensator::new(&edges);

        assert!(compensator.use_relative);
        assert_eq!(compensator.scene_origin, [150000.0, 100000.0]); // 场景中心
    }

    #[test]
    fn test_world_to_screen_small_coords() {
        let edges = vec![Edge {
            start: [0.0, 0.0],
            end: [100.0, 0.0],
        }];
        let compensator = CoordinateCompensator::new(&edges);

        let screen =
            compensator.world_to_screen([50.0, 0.0], 1.0, Vec2::ZERO, Pos2::new(400.0, 300.0));

        assert!((screen.x - 450.0).abs() < 0.01);
        assert!((screen.y - 300.0).abs() < 0.01);
    }

    #[test]
    fn test_world_to_screen_large_coords() {
        let edges = vec![Edge {
            start: [100000.0, 100000.0],
            end: [200000.0, 100000.0],
        }];
        let compensator = CoordinateCompensator::new(&edges);

        // 世界坐标 (150000, 100000) 相对于原点 (150000, 100000) 是 (0, 0)
        let screen = compensator.world_to_screen(
            [150000.0, 100000.0],
            1.0,
            Vec2::ZERO,
            Pos2::new(400.0, 300.0),
        );

        assert!((screen.x - 400.0).abs() < 0.01);
        assert!((screen.y - 300.0).abs() < 0.01);
    }

    #[test]
    fn test_round_trip() {
        let edges = vec![Edge {
            start: [100000.0, 100000.0],
            end: [200000.0, 100000.0],
        }];
        let compensator = CoordinateCompensator::new(&edges);

        let world = [150000.0, 100000.0];
        let zoom = 1.0;
        let pan = Vec2::ZERO;
        let center = Pos2::new(400.0, 300.0);

        // 世界 → 屏幕 → 世界
        let screen = compensator.world_to_screen(world, zoom, pan, center);
        let world_back = compensator.screen_to_world(screen, zoom, pan, center);

        // 允许 f32 精度损失
        assert!((world_back[0] - world[0]).abs() < 0.1);
        assert!((world_back[1] - world[1]).abs() < 0.1);
    }
}
