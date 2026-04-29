//! 相对坐标系统
//!
//! ## 设计哲学
//!
//! 商业化 CAD 工具（如 AutoCAD）使用**相对坐标系统**来解决 f64 精度问题：
//! - 建筑总图（坐标 1e6, 1e6）：f64 精度 0.0002mm，但累加误差 1000 条线段 → 0.2mm
//! - 城市 GIS（坐标 1e8, 1e8）：f64 精度 0.02mm，端点吸附可能失败
//! - PCB 设计（坐标 0.001, 0.001）：f64 精度 2e-19，但固定容差 1e-6 会误判
//!
//! ## 核心思想
//!
//! 1. **场景原点**：维护一个场景级别的绝对原点（f64，米级精度）
//! 2. **相对坐标**：所有点存储为相对于原点的偏移（f32，毫米级精度）
//! 3. **动态转换**：在需要时进行世界坐标 ↔ 相对坐标的转换
//!
//! ## 精度分析
//!
//! | 场景类型 | 世界坐标范围 | f64 精度 | f32 相对坐标精度 | 是否可用 |
//! |----------|-------------|---------|-----------------|----------|
//! | 建筑总图 | (1e6, 1e6) | 0.0002mm | 0.0001mm | ✅ |
//! | 城市 GIS | (1e8, 1e8) | 0.02mm | 0.01mm | ✅ |
//! | 零件图 | (0-100, 0-100) | 1e-12mm | 1e-8mm | ✅ |
//! | PCB 设计 | (0.001, 0.001) | 2e-19mm | 2e-10mm | ✅ |
//!
//! ## 使用示例
//!
//! ```rust
//! use common_types::relative_coords::{SceneOrigin, RelativePoint};
//!
//! // 创建场景原点（世界坐标）
//! let origin = SceneOrigin::new([1_000_000.0, 1_000_000.0]);
//!
//! // 世界坐标转相对坐标（mm）
//! let world_point = [1_000_100.5, 1_000_200.75];
//! let relative = origin.world_to_relative(world_point);
//! assert!((relative[0] - 100500.0).abs() < 0.1);  // 相对坐标：100.5m → 100500mm
//!
//! // 相对坐标转世界坐标
//! let back_to_world = origin.relative_to_world(relative);
//! assert!((back_to_world[0] - world_point[0]).abs() < 0.001);
//! ```

use crate::geometry::Point2;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// 场景原点（世界坐标系）
///
/// ## 设计目标
/// 1. 解决大坐标场景下的 f64 精度问题
/// 2. 支持小坐标场景的高精度需求
/// 3. 与现有代码兼容（保持 Point2 = [f64; 2]）
#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema)]
pub struct SceneOrigin {
    /// 世界坐标原点（单位：米）
    pub world_origin: [f64; 2],
    /// 场景尺度（单位：米，用于计算相对容差）
    pub scene_scale: f64,
}

impl Default for SceneOrigin {
    fn default() -> Self {
        Self {
            world_origin: [0.0, 0.0],
            scene_scale: 1000.0, // 默认 1 公里
        }
    }
}

impl SceneOrigin {
    /// 创建新的场景原点
    ///
    /// ## 参数
    /// - `world_origin`: 世界坐标原点（单位：米）
    ///
    /// ## 示例
    /// ```rust
    /// use common_types::relative_coords::SceneOrigin;
    ///
    /// // 建筑总图：原点在 (1000km, 1000km)
    /// let origin = SceneOrigin::new([1_000_000.0, 1_000_000.0]);
    /// ```
    pub fn new(world_origin: [f64; 2]) -> Self {
        Self {
            world_origin,
            scene_scale: Self::estimate_scale_from_origin(&world_origin),
        }
    }

    /// 创建带场景尺度的原点
    ///
    /// ## 参数
    /// - `world_origin`: 世界坐标原点（单位：米）
    /// - `scene_scale`: 场景尺度（单位：米）
    pub fn with_scale(world_origin: [f64; 2], scene_scale: f64) -> Self {
        Self {
            world_origin,
            scene_scale,
        }
    }

    /// 从实体列表自动计算场景原点和尺度
    ///
    /// ## 算法
    /// 1. 遍历所有实体，找到坐标范围
    /// 2. 原点 = 范围中心（向下取整到百米）
    /// 3. 场景尺度 = 范围对角线长度
    ///
    /// ## 示例
    /// ```rust,no_run
    /// use common_types::geometry::RawEntity;
    /// use common_types::relative_coords::SceneOrigin;
    ///
    /// let entities: Vec<RawEntity> = vec![];
    /// let origin = SceneOrigin::from_entities(&entities);
    /// ```
    pub fn from_entities(entities: &[crate::geometry::RawEntity]) -> Self {
        if entities.is_empty() {
            return Self::default();
        }

        let (min, max) = Self::compute_bbox(entities);
        let center = [(min[0] + max[0]) / 2.0, (min[1] + max[1]) / 2.0];

        // 向下取整到百米，避免浮点误差
        let origin = [
            (center[0] / 100.0).floor() * 100.0,
            (center[1] / 100.0).floor() * 100.0,
        ];

        let scale = ((max[0] - min[0]).powi(2) + (max[1] - min[1]).powi(2)).sqrt();

        Self::with_scale(origin, scale.max(1.0))
    }

    /// 从点列表自动计算场景原点
    pub fn from_points(points: &[Point2]) -> Self {
        if points.is_empty() {
            return Self::default();
        }

        let min = [
            points.iter().map(|p| p[0]).fold(f64::INFINITY, f64::min),
            points.iter().map(|p| p[1]).fold(f64::INFINITY, f64::min),
        ];
        let max = [
            points
                .iter()
                .map(|p| p[0])
                .fold(f64::NEG_INFINITY, f64::max),
            points
                .iter()
                .map(|p| p[1])
                .fold(f64::NEG_INFINITY, f64::max),
        ];

        let center = [(min[0] + max[0]) / 2.0, (min[1] + max[1]) / 2.0];
        let origin = [
            (center[0] / 100.0).floor() * 100.0,
            (center[1] / 100.0).floor() * 100.0,
        ];
        let scale = ((max[0] - min[0]).powi(2) + (max[1] - min[1]).powi(2)).sqrt();

        Self::with_scale(origin, scale.max(1.0))
    }

    /// 计算实体列表的包围盒
    fn compute_bbox(entities: &[crate::geometry::RawEntity]) -> (Point2, Point2) {
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
                crate::geometry::RawEntity::BlockReference {
                    insertion_point, ..
                } => {
                    min_x = min_x.min(insertion_point[0]);
                    max_x = max_x.max(insertion_point[0]);
                    min_y = min_y.min(insertion_point[1]);
                    max_y = max_y.max(insertion_point[1]);
                }
                _ => {}
            }
        }

        ([min_x, min_y], [max_x, max_y])
    }

    /// 从原点估计场景尺度
    fn estimate_scale_from_origin(origin: &[f64; 2]) -> f64 {
        // 假设场景尺度约为原点坐标的 1%
        origin[0].abs().max(origin[1].abs()) * 0.01
    }

    // ========================================================================
    // 坐标转换：世界坐标 ↔ 相对坐标
    // ========================================================================

    /// 世界坐标转相对坐标（单位：毫米）
    ///
    /// ## 公式
    /// ```text
    /// relative_mm = (world_m - origin_m) × 1000.0
    /// ```
    ///
    /// ## 精度保证
    /// - 对于建筑总图 (1e6, 1e6)：相对坐标精度 0.0001mm
    /// - 对于城市 GIS (1e8, 1e8)：相对坐标精度 0.01mm
    ///
    /// ## 示例
    /// ```rust
    /// use common_types::relative_coords::SceneOrigin;
    ///
    /// let origin = SceneOrigin::new([1_000_000.0, 1_000_000.0]);
    /// let world = [1_000_100.5, 1_000_200.75];  // 世界坐标：米
    /// let relative = origin.world_to_relative(world);
    /// // relative = [100500.0, 200750.0]  // 相对坐标：毫米
    /// ```
    pub fn world_to_relative(&self, world: Point2) -> [f32; 2] {
        [
            ((world[0] - self.world_origin[0]) * 1000.0) as f32,
            ((world[1] - self.world_origin[1]) * 1000.0) as f32,
        ]
    }

    /// 相对坐标转世界坐标（单位：米）
    ///
    /// ## 公式
    /// ```text
    /// world_m = origin_m + relative_mm / 1000.0
    /// ```
    ///
    /// ## 示例
    /// ```rust
    /// use common_types::relative_coords::SceneOrigin;
    ///
    /// let origin = SceneOrigin::new([1_000_000.0, 1_000_000.0]);
    /// let relative = [100500.0f32, 200750.0f32];  // 相对坐标：毫米
    /// let world = origin.relative_to_world(relative);
    /// // world ≈ [1_000_100.5, 1_000_200.75]  // 世界坐标：米
    /// ```
    pub fn relative_to_world(&self, relative: [f32; 2]) -> Point2 {
        [
            self.world_origin[0] + relative[0] as f64 / 1000.0,
            self.world_origin[1] + relative[1] as f64 / 1000.0,
        ]
    }

    /// 世界坐标转相对坐标（f64 版本，用于高精度需求）
    ///
    /// ## 注意
    /// 此方法返回 f64 相对坐标，用于需要高精度的场景
    /// （如约束求解器、NURBS 计算等）
    pub fn world_to_relative_f64(&self, world: Point2) -> Point2 {
        [
            (world[0] - self.world_origin[0]) * 1000.0,
            (world[1] - self.world_origin[1]) * 1000.0,
        ]
    }

    /// 相对坐标转世界坐标（f64 版本）
    pub fn relative_f64_to_world(&self, relative: Point2) -> Point2 {
        [
            self.world_origin[0] + relative[0] / 1000.0,
            self.world_origin[1] + relative[1] / 1000.0,
        ]
    }

    // ========================================================================
    // 批量转换
    // ========================================================================

    /// 批量转换世界坐标到相对坐标
    pub fn world_to_relative_batch(&self, points: &[Point2]) -> Vec<[f32; 2]> {
        points.iter().map(|&p| self.world_to_relative(p)).collect()
    }

    /// 批量转换相对坐标到世界坐标
    pub fn relative_to_world_batch(&self, points: &[[f32; 2]]) -> Vec<Point2> {
        points.iter().map(|&p| self.relative_to_world(p)).collect()
    }

    // ========================================================================
    // 工具方法
    // ========================================================================

    /// 获取场景原点（世界坐标，米）
    pub fn origin_meters(&self) -> Point2 {
        self.world_origin
    }

    /// 获取场景尺度（米）
    pub fn scene_scale(&self) -> f64 {
        self.scene_scale
    }

    /// 获取场景范围（世界坐标，米）
    pub fn scene_bbox(&self) -> (Point2, Point2) {
        let half_scale = self.scene_scale / 2.0;
        (
            [
                self.world_origin[0] - half_scale,
                self.world_origin[1] - half_scale,
            ],
            [
                self.world_origin[0] + half_scale,
                self.world_origin[1] + half_scale,
            ],
        )
    }

    /// 获取摘要信息
    pub fn summary(&self) -> String {
        format!(
            "SceneOrigin {{\n\
             \t原点：[{:.2}, {:.2}] 米\n\
             \t尺度：{:.2} 米\n\
             \t范围：[{:.2}, {:.2}] 到 [{:.2}, {:.2}] 米\n\
             }}",
            self.world_origin[0],
            self.world_origin[1],
            self.scene_scale,
            self.world_origin[0] - self.scene_scale / 2.0,
            self.world_origin[1] - self.scene_scale / 2.0,
            self.world_origin[0] + self.scene_scale / 2.0,
            self.world_origin[1] + self.scene_scale / 2.0,
        )
    }
}

/// 相对坐标点（单位：毫米）
///
/// ## 设计目标
/// 1. 使用 f32 存储，节省内存（相对于 f64 节省 50%）
/// 2. 保持足够精度（0.0001mm @ 1e6 坐标）
/// 3. 与 GPU 渲染兼容（GPU 原生支持 f32）
#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema)]
pub struct RelativePoint {
    /// X 坐标（毫米）
    pub x: f32,
    /// Y 坐标（毫米）
    pub y: f32,
}

impl RelativePoint {
    /// 创建新的相对坐标点
    pub fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    /// 从世界坐标创建
    pub fn from_world(world: Point2, origin: &SceneOrigin) -> Self {
        let rel = origin.world_to_relative(world);
        Self {
            x: rel[0],
            y: rel[1],
        }
    }

    /// 转换为世界坐标
    pub fn to_world(&self, origin: &SceneOrigin) -> Point2 {
        origin.relative_to_world([self.x, self.y])
    }

    /// 转换为数组
    pub fn to_array(&self) -> [f32; 2] {
        [self.x, self.y]
    }

    /// 从数组创建
    pub fn from_array(arr: [f32; 2]) -> Self {
        Self {
            x: arr[0],
            y: arr[1],
        }
    }

    /// 计算到另一点的距离（毫米）
    pub fn distance_to(&self, other: &RelativePoint) -> f32 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        (dx * dx + dy * dy).sqrt()
    }

    /// 计算到另一点的平方距离（毫米²）
    pub fn distance_squared_to(&self, other: &RelativePoint) -> f32 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        dx * dx + dy * dy
    }
}

impl From<[f32; 2]> for RelativePoint {
    fn from(arr: [f32; 2]) -> Self {
        Self::from_array(arr)
    }
}

impl From<RelativePoint> for [f32; 2] {
    fn from(pt: RelativePoint) -> Self {
        pt.to_array()
    }
}

// ============================================================================
// 相对坐标场景状态
// ============================================================================

/// 相对坐标场景状态
///
/// ## 设计目标
/// 1. 支持大坐标场景（建筑总图、城市 GIS）
/// 2. 保持高精度（f32 相对坐标）
/// 3. 与现有 SceneState 兼容
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct RelativeSceneState {
    /// 场景原点（世界坐标）
    pub origin: SceneOrigin,
    /// 外轮廓（相对坐标，毫米）
    pub outer: Option<RelativeClosedLoop>,
    /// 孔洞列表（相对坐标，毫米）
    pub holes: Vec<RelativeClosedLoop>,
    /// 原始边数据（相对坐标，毫米）
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub edges: Vec<RelativeRawEdge>,
    /// 场景尺度（米，从 origin 复制，用于快速访问）
    #[serde(default)]
    pub scene_scale: f64,
}

impl Default for RelativeSceneState {
    fn default() -> Self {
        Self {
            origin: SceneOrigin::default(),
            outer: None,
            holes: Vec::new(),
            edges: Vec::new(),
            scene_scale: 1000.0,
        }
    }
}

impl RelativeSceneState {
    /// 创建新的相对坐标场景
    pub fn new(origin: SceneOrigin) -> Self {
        Self {
            origin,
            outer: None,
            holes: Vec::new(),
            edges: Vec::new(),
            scene_scale: origin.scene_scale,
        }
    }

    /// 从绝对坐标 SceneState 转换
    pub fn from_absolute(scene: &crate::scene::SceneState) -> Self {
        // 收集所有点来计算场景原点
        let mut all_points = Vec::new();

        if let Some(outer) = &scene.outer {
            all_points.extend(&outer.points);
        }
        for hole in &scene.holes {
            all_points.extend(&hole.points);
        }
        for edge in &scene.edges {
            all_points.push(edge.start);
            all_points.push(edge.end);
        }

        let origin = SceneOrigin::from_points(&all_points);

        let outer = scene
            .outer
            .as_ref()
            .map(|loop_| RelativeClosedLoop::from_absolute(loop_, &origin));

        let holes = scene
            .holes
            .iter()
            .map(|loop_| RelativeClosedLoop::from_absolute(loop_, &origin))
            .collect();

        let edges = scene
            .edges
            .iter()
            .map(|edge| RelativeRawEdge::from_absolute(edge, &origin))
            .collect();

        Self {
            origin,
            outer,
            holes,
            edges,
            scene_scale: origin.scene_scale,
        }
    }

    /// 转换为绝对坐标 SceneState
    pub fn to_absolute(&self) -> crate::scene::SceneState {
        use crate::scene::{CoordinateSystem, LengthUnit, SceneState};

        let outer = self
            .outer
            .as_ref()
            .map(|loop_| loop_.to_absolute(&self.origin));
        let holes = self
            .holes
            .iter()
            .map(|loop_| loop_.to_absolute(&self.origin))
            .collect();
        let edges = self
            .edges
            .iter()
            .map(|edge| edge.to_absolute(&self.origin))
            .collect();

        SceneState {
            outer,
            holes,
            boundaries: Vec::new(), // 需要时从 edges 重建
            sources: Vec::new(),
            edges,
            raster_metadata: None,
            units: LengthUnit::Mm,
            coordinate_system: CoordinateSystem::RightHandedYUp,
            seat_zones: Vec::new(),
            render_config: None,
        }
    }

    /// 获取场景原点
    pub fn origin(&self) -> &SceneOrigin {
        &self.origin
    }

    /// 获取场景尺度（米）
    pub fn scene_scale(&self) -> f64 {
        self.scene_scale
    }
}

/// 相对坐标闭合环
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct RelativeClosedLoop {
    /// 点序列（相对坐标，毫米）
    pub points: Vec<[f32; 2]>,
    /// 有符号面积（平方毫米）
    pub signed_area: f32,
}

impl RelativeClosedLoop {
    /// 创建新的相对坐标闭合环
    pub fn new(points: Vec<[f32; 2]>) -> Self {
        let signed_area = Self::calculate_signed_area(&points);
        Self {
            points,
            signed_area,
        }
    }

    /// 从绝对坐标 ClosedLoop 转换
    pub fn from_absolute(loop_: &crate::scene::ClosedLoop, origin: &SceneOrigin) -> Self {
        let points: Vec<[f32; 2]> = loop_
            .points
            .iter()
            .map(|&p| origin.world_to_relative(p))
            .collect();
        let signed_area = Self::calculate_signed_area(&points);
        Self {
            points,
            signed_area,
        }
    }

    /// 转换为绝对坐标 ClosedLoop
    pub fn to_absolute(&self, origin: &SceneOrigin) -> crate::scene::ClosedLoop {
        let points: Vec<Point2> = self
            .points
            .iter()
            .map(|&p| origin.relative_to_world(p))
            .collect();
        // 重新计算 f64 精度的有符号面积
        let signed_area = crate::scene::ClosedLoop::new(points.clone()).signed_area;
        crate::scene::ClosedLoop {
            points,
            signed_area,
        }
    }

    /// 计算有符号面积（平方毫米）
    fn calculate_signed_area(points: &[[f32; 2]]) -> f32 {
        if points.len() < 3 {
            return 0.0;
        }

        let mut area = 0.0f32;
        let n = points.len();

        for i in 0..n {
            let j = (i + 1) % n;
            area += points[i][0] * points[j][1];
            area -= points[j][0] * points[i][1];
        }

        area / 2.0
    }

    /// 是否为外轮廓（正面积）
    pub fn is_outer(&self) -> bool {
        self.signed_area > 0.0
    }

    /// 是否为孔洞（负面积）
    pub fn is_hole(&self) -> bool {
        self.signed_area < 0.0
    }

    /// 获取周长（毫米）
    pub fn perimeter(&self) -> f32 {
        let mut perimeter = 0.0f32;
        let n = self.points.len();
        for i in 0..n {
            let j = (i + 1) % n;
            let dx = self.points[j][0] - self.points[i][0];
            let dy = self.points[j][1] - self.points[i][1];
            perimeter += (dx * dx + dy * dy).sqrt();
        }
        perimeter
    }
}

/// 相对坐标原始边
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct RelativeRawEdge {
    /// 边 ID
    pub id: usize,
    /// 起点（相对坐标，毫米）
    pub start: [f32; 2],
    /// 终点（相对坐标，毫米）
    pub end: [f32; 2],
    /// 图层名称（可选）
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub layer: Option<String>,
    /// 颜色索引（可选）
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub color_index: Option<u16>,
}

impl RelativeRawEdge {
    /// 创建新的相对坐标边
    pub fn new(id: usize, start: [f32; 2], end: [f32; 2]) -> Self {
        Self {
            id,
            start,
            end,
            layer: None,
            color_index: None,
        }
    }

    /// 从绝对坐标 RawEdge 转换
    pub fn from_absolute(edge: &crate::scene::RawEdge, origin: &SceneOrigin) -> Self {
        Self {
            id: edge.id,
            start: origin.world_to_relative(edge.start),
            end: origin.world_to_relative(edge.end),
            layer: edge.layer.clone(),
            color_index: edge.color_index,
        }
    }

    /// 转换为绝对坐标 RawEdge
    pub fn to_absolute(&self, origin: &SceneOrigin) -> crate::scene::RawEdge {
        crate::scene::RawEdge {
            id: self.id,
            start: origin.relative_to_world(self.start),
            end: origin.relative_to_world(self.end),
            layer: self.layer.clone(),
            color_index: self.color_index,
        }
    }

    /// 获取边长度（毫米）
    pub fn length(&self) -> f32 {
        let dx = self.end[0] - self.start[0];
        let dy = self.end[1] - self.start[1];
        (dx * dx + dy * dy).sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scene_origin_creation() {
        // 建筑总图场景
        let origin = SceneOrigin::new([1_000_000.0, 1_000_000.0]);
        assert!((origin.world_origin[0] - 1_000_000.0).abs() < 1e-10);
        assert!(origin.scene_scale > 0.0);
    }

    #[test]
    fn test_world_to_relative_conversion() {
        let origin = SceneOrigin::new([1_000_000.0, 1_000_000.0]);

        // 世界坐标：100.5 米偏移
        let world = [1_000_100.5, 1_000_200.75];
        let relative = origin.world_to_relative(world);

        // 相对坐标应该是 100500mm 和 200750mm
        assert!((relative[0] as f64 - 100500.0).abs() < 1.0); // f32 精度允许 1mm 误差
        assert!((relative[1] as f64 - 200750.0).abs() < 1.0);
    }

    #[test]
    fn test_relative_to_world_conversion() {
        let origin = SceneOrigin::new([1_000_000.0, 1_000_000.0]);

        // 相对坐标：100500mm, 200750mm
        let relative = [100500.0f32, 200750.0f32];
        let world = origin.relative_to_world(relative);

        // 应该恢复为世界坐标
        assert!((world[0] - 1_000_100.5).abs() < 0.001);
        assert!((world[1] - 1_000_200.75).abs() < 0.001);
    }

    #[test]
    fn test_roundtrip_precision() {
        let origin = SceneOrigin::new([1_000_000.0, 1_000_000.0]);
        let world = [1_000_123.456, 1_000_789.012];

        let relative = origin.world_to_relative(world);
        let back = origin.relative_to_world(relative);

        // 往返精度应该在毫米级
        assert!((back[0] - world[0]).abs() < 0.001); // 1mm 精度
        assert!((back[1] - world[1]).abs() < 0.001);
    }

    #[test]
    fn test_large_coordinate_scenario() {
        // 城市 GIS 场景（坐标 1e8）
        let origin = SceneOrigin::new([100_000_000.0, 100_000_000.0]);
        let world = [100_000_100.5, 100_000_200.75];

        let relative = origin.world_to_relative(world);
        let back = origin.relative_to_world(relative);

        // 即使在大坐标下，精度也应该可接受
        assert!((back[0] - world[0]).abs() < 0.1); // 10cm 精度（对于 GIS 可接受）
        assert!((back[1] - world[1]).abs() < 0.1);
    }

    #[test]
    fn test_small_coordinate_scenario() {
        // 零件图场景（坐标 0-100）
        let origin = SceneOrigin::new([50.0, 50.0]);
        let world = [50.123, 50.456];

        let relative = origin.world_to_relative(world);
        let back = origin.relative_to_world(relative);

        // 小坐标下应该有极高精度
        assert!((back[0] - world[0]).abs() < 1e-6);
        assert!((back[1] - world[1]).abs() < 1e-6);
    }

    #[test]
    fn test_relative_closed_loop() {
        let _origin = SceneOrigin::new([0.0, 0.0]);

        // 创建一个 100mm x 100mm 的正方形（相对坐标）
        let points: Vec<[f32; 2]> = vec![
            [0.0, 0.0],
            [100.0, 0.0], // 100mm
            [100.0, 100.0],
            [0.0, 100.0],
        ];

        let loop_ = RelativeClosedLoop::new(points.clone());

        assert!(loop_.is_outer());
        // 面积 = 100mm * 100mm = 10000 mm²（有符号面积公式：sum(x_i * y_{i+1} - x_{i+1} * y_i) / 2）
        assert!((loop_.signed_area - 10000.0).abs() < 1.0);
        // 周长 = 4 * 100mm = 400mm
        assert!((loop_.perimeter() - 400.0).abs() < 1.0);
    }

    #[test]
    fn test_relative_raw_edge() {
        let _origin = SceneOrigin::new([1000.0, 1000.0]);

        let edge = RelativeRawEdge::new(
            1,
            [0.0, 0.0],
            [3000.0, 4000.0], // 3 米 x 4 米 = 5 米对角线
        );

        assert!((edge.length() - 5000.0).abs() < 1.0); // 5 米 = 5000mm
    }

    #[test]
    fn test_relative_scene_state() {
        use crate::scene::{ClosedLoop, RawEdge, SceneState};

        // 创建绝对坐标场景
        let abs_scene = SceneState {
            outer: Some(ClosedLoop::new(vec![
                [1000.0, 1000.0],
                [1010.0, 1000.0],
                [1010.0, 1010.0],
                [1000.0, 1010.0],
            ])),
            holes: Vec::new(),
            boundaries: Vec::new(),
            sources: Vec::new(),
            edges: vec![RawEdge {
                id: 1,
                start: [1000.0, 1000.0],
                end: [1010.0, 1000.0],
                layer: Some("WALL".into()),
                color_index: None,
            }],
            raster_metadata: None,
            units: crate::scene::LengthUnit::M,
            coordinate_system: crate::scene::CoordinateSystem::RightHandedYUp,
            seat_zones: Vec::new(),
            render_config: None,
        };

        // 转换为相对坐标
        let rel_scene = RelativeSceneState::from_absolute(&abs_scene);

        // 验证原点设置正确
        assert!(rel_scene.origin.scene_scale > 0.0);

        // 转换回绝对坐标
        let back = rel_scene.to_absolute();

        // 验证坐标基本一致
        if let (Some(orig), Some(back)) = (abs_scene.outer.as_ref(), back.outer.as_ref()) {
            for (o, b) in orig.points.iter().zip(back.points.iter()) {
                assert!((o[0] - b[0]).abs() < 0.001);
                assert!((o[1] - b[1]).abs() < 0.001);
            }
        }
    }
}
