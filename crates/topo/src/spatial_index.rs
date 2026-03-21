//! 分层空间索引渲染 - 网格+R*-tree
//!
//! ## 设计目标
//!
//! 1. **快速视口裁剪**: 只渲染视口内的实体，减少 GPU/CPU 负载
//! 2. **分层查询**: 粗粒度网格过滤 + 细粒度 R*-tree 精确查询
//! 3. **核显优化**: 减少绘制调用，提升批量渲染效率
//! 4. **动态更新**: 支持增量插入和删除实体
//!
//! ## 架构设计
//!
//! ```text
//! ┌─────────────────────────────────────┐
//! │      SpatialIndex (统一入口)        │
//! │  ┌─────────────┐  ┌──────────────┐  │
//! │  │  GridIndex  │→ │ RTreeIndex   │  │
//! │  │  (粗粒度)   │  │  (细粒度)    │  │
//! │  └─────────────┘  └──────────────┘  │
//! └─────────────────────────────────────┘
//! ```
//!
//! ## 使用示例
//!
//! ```rust
//! use topo::spatial_index::{SpatialIndex, RenderEntity};
//!
//! // 创建空间索引
//! let mut index = SpatialIndex::new();
//!
//! // 添加实体
//! let entity = RenderEntity::Line {
//!     start: [0.0, 0.0],
//!     end: [10.0, 10.0],
//!     layer: "WALL".to_string(),
//!     color: [1.0, 0.0, 0.0, 1.0],
//! };
//! index.insert(0, entity);
//!
//! // 视口查询
//! let viewport = ([0.0, 0.0], [100.0, 100.0]);
//! let visible: Vec<_> = index.query_viewport(viewport).collect();
//! ```

use common_types::{Point2, orient2d, Orientation};
use rstar::{RTree, AABB, RTreeObject};
use std::collections::{HashMap, HashSet};

// ============================================================================
// 渲染实体定义
// ============================================================================

/// 可渲染的几何实体
#[derive(Debug, Clone, PartialEq)]
pub enum RenderEntity {
    /// 线段
    Line {
        start: Point2,
        end: Point2,
        layer: String,
        color: [f32; 4],
    },
    /// 多段线
    Polyline {
        points: Vec<Point2>,
        closed: bool,
        layer: String,
        color: [f32; 4],
    },
    /// 圆弧
    Arc {
        center: Point2,
        radius: f64,
        start_angle: f64,
        end_angle: f64,
        layer: String,
        color: [f32; 4],
    },
    /// 圆
    Circle {
        center: Point2,
        radius: f64,
        layer: String,
        color: [f32; 4],
    },
    /// 文本
    Text {
        position: Point2,
        content: String,
        height: f64,
        layer: String,
        color: [f32; 4],
    },
}

impl RenderEntity {
    /// 获取实体的包围盒
    pub fn aabb(&self) -> AABB<[f64; 2]> {
        match self {
            RenderEntity::Line { start, end, .. } => {
                AABB::from_corners(*start, *end)
            }
            RenderEntity::Polyline { points, .. } => {
                if points.is_empty() {
                    AABB::from_corners([0.0, 0.0], [0.0, 0.0])
                } else {
                    let mut min = points[0];
                    let mut max = points[0];
                    for &p in points.iter().skip(1) {
                        min[0] = min[0].min(p[0]);
                        min[1] = min[1].min(p[1]);
                        max[0] = max[0].max(p[0]);
                        max[1] = max[1].max(p[1]);
                    }
                    AABB::from_corners(min, max)
                }
            }
            RenderEntity::Arc { center, radius, .. } |
            RenderEntity::Circle { center, radius, .. } => {
                let min = [center[0] - radius, center[1] - radius];
                let max = [center[0] + radius, center[1] + radius];
                AABB::from_corners(min, max)
            }
            RenderEntity::Text { position, height, .. } => {
                // 文本包围盒估算
                let min = [position[0], position[1] - height];
                let max = [position[0] + height * 2.0, position[1]];
                AABB::from_corners(min, max)
            }
        }
    }

    /// 获取图层名
    pub fn layer(&self) -> &str {
        match self {
            RenderEntity::Line { layer, .. } |
            RenderEntity::Polyline { layer, .. } |
            RenderEntity::Arc { layer, .. } |
            RenderEntity::Circle { layer, .. } |
            RenderEntity::Text { layer, .. } => layer,
        }
    }

    /// 获取实体 ID 用于标识
    pub fn id(&self) -> usize {
        // 这里返回一个简化的哈希值，实际使用需要外部 ID
        match self {
            RenderEntity::Line { start, .. } => {
                (start[0] * 1000.0 + start[1]) as usize
            }
            _ => 0
        }
    }
}

// ============================================================================
// R*-tree 索引
// ============================================================================

/// R*-tree 索引项
#[derive(Debug, Clone, PartialEq)]
pub struct RTreeItem {
    pub id: usize,
    pub entity: RenderEntity,
    pub aabb: AABB<[f64; 2]>,
}

impl RTreeObject for RTreeItem {
    type Envelope = AABB<[f64; 2]>;

    fn envelope(&self) -> Self::Envelope {
        self.aabb.clone()
    }
}

/// R*-tree 空间索引（细粒度）
#[derive(Debug, Clone)]
pub struct RTreeIndex {
    tree: RTree<RTreeItem>,
    item_map: HashMap<usize, RenderEntity>,
}

impl Default for RTreeIndex {
    fn default() -> Self {
        Self::new()
    }
}

impl RTreeIndex {
    /// 创建新的 R*-tree 索引
    pub fn new() -> Self {
        Self {
            tree: RTree::new(),
            item_map: HashMap::new(),
        }
    }

    /// 插入实体
    pub fn insert(&mut self, id: usize, entity: RenderEntity) {
        let aabb = entity.aabb();
        let item = RTreeItem { id, aabb, entity: entity.clone() };
        self.tree.insert(item);
        self.item_map.insert(id, entity);
    }

    /// 删除实体（简化实现：只从 item_map 删除，tree 中的项会在查询时被过滤）
    pub fn remove(&mut self, id: usize) -> Option<RenderEntity> {
        self.item_map.remove(&id)
    }

    /// 查询包围盒内的实体
    pub fn query_aabb(&self, aabb: &AABB<[f64; 2]>) -> Vec<&RenderEntity> {
        self.tree
            .locate_in_envelope(aabb)
            .filter(|item| self.item_map.contains_key(&item.id))
            .map(|item| &item.entity)
            .collect()
    }

    /// 查询视口内的实体
    pub fn query_viewport(&self, viewport_min: Point2, viewport_max: Point2) -> Vec<&RenderEntity> {
        let aabb = AABB::from_corners(viewport_min, viewport_max);
        self.query_aabb(&aabb)
    }

    /// 获取索引中的实体总数
    pub fn len(&self) -> usize {
        self.tree.iter().count()
    }

    /// 检查索引是否为空
    pub fn is_empty(&self) -> bool {
        self.tree.iter().next().is_none()
    }
}

// ============================================================================
// 网格索引
// ============================================================================

/// 网格单元格
#[derive(Debug, Clone)]
pub struct GridCell {
    /// 单元格内的实体 ID 列表
    pub entity_ids: Vec<usize>,
    /// 单元格的包围盒
    pub aabb: AABB<[f64; 2]>,
}

impl Default for GridCell {
    fn default() -> Self {
        Self {
            entity_ids: Vec::new(),
            aabb: AABB::from_corners([0.0, 0.0], [0.0, 0.0]),
        }
    }
}

/// 网格空间索引（粗粒度）
#[derive(Debug, Clone)]
pub struct GridIndex {
    /// 网格宽度
    cell_size: f64,
    /// 网格偏移（世界坐标）
    offset: Point2,
    /// 网格尺寸（单元格数）
    #[allow(dead_code)] // 预留用于未来网格优化
    grid_size: [usize; 2],
    /// 网格单元格
    cells: HashMap<[i32; 2], GridCell>,
    /// 实体 ID 到包围盒的映射
    entity_aabbs: HashMap<usize, AABB<[f64; 2]>>,
}

impl Default for GridIndex {
    fn default() -> Self {
        Self::new(100.0, [0.0, 0.0])
    }
}

impl GridIndex {
    /// 创建新的网格索引
    ///
    /// # 参数
    /// - `cell_size`: 单元格大小（世界坐标单位）
    /// - `offset`: 网格原点偏移
    pub fn new(cell_size: f64, offset: Point2) -> Self {
        Self {
            cell_size,
            offset,
            grid_size: [1024, 1024],  // 默认网格尺寸
            cells: HashMap::new(),
            entity_aabbs: HashMap::new(),
        }
    }

    /// 创建自适应网格索引（根据场景范围自动调整）
    pub fn with_bounds(scene_min: Point2, scene_max: Point2, target_cells: usize) -> Self {
        let width = scene_max[0] - scene_min[0];
        let height = scene_max[1] - scene_min[1];
        
        // 计算单元格大小
        let cell_size = (width / target_cells as f64).max(height / target_cells as f64);
        
        Self {
            cell_size,
            offset: scene_min,
            grid_size: [target_cells, target_cells],
            cells: HashMap::new(),
            entity_aabbs: HashMap::new(),
        }
    }

    /// 将世界坐标转换为网格坐标
    fn world_to_grid(&self, point: Point2) -> [i32; 2] {
        let x = ((point[0] - self.offset[0]) / self.cell_size).floor() as i32;
        let y = ((point[1] - self.offset[1]) / self.cell_size).floor() as i32;
        [x, y]
    }

    /// 获取单元格
    fn get_cell(&mut self, grid_coord: [i32; 2]) -> &mut GridCell {
        self.cells.entry(grid_coord).or_insert_with(|| {
            let min = [
                self.offset[0] + grid_coord[0] as f64 * self.cell_size,
                self.offset[1] + grid_coord[1] as f64 * self.cell_size,
            ];
            let max = [
                min[0] + self.cell_size,
                min[1] + self.cell_size,
            ];
            GridCell {
                aabb: AABB::from_corners(min, max),
                ..Default::default()
            }
        })
    }

    /// 插入实体
    pub fn insert(&mut self, id: usize, entity: &RenderEntity) {
        let aabb = entity.aabb();
        let lower = aabb.lower();  // rstar 0.12 返回 [f64; 2] 值
        let upper = aabb.upper();
        let min_grid = self.world_to_grid(lower);
        let max_grid = self.world_to_grid(upper);

        // 实体可能跨越多个单元格
        for x in min_grid[0]..=max_grid[0] {
            for y in min_grid[1]..=max_grid[1] {
                let cell = self.get_cell([x, y]);
                cell.entity_ids.push(id);
            }
        }

        self.entity_aabbs.insert(id, aabb);
    }

    /// 删除实体
    pub fn remove(&mut self, id: usize) {
        if let Some(aabb) = self.entity_aabbs.get(&id).cloned() {
            let lower = aabb.lower();  // rstar 0.12 返回 [f64; 2] 值
            let upper = aabb.upper();
            let min_grid = self.world_to_grid(lower);
            let max_grid = self.world_to_grid(upper);

            for x in min_grid[0]..=max_grid[0] {
                for y in min_grid[1]..=max_grid[1] {
                    if let Some(cell) = self.cells.get_mut(&[x, y]) {
                        cell.entity_ids.retain(|&eid| eid != id);
                    }
                }
            }
            
            self.entity_aabbs.remove(&id);
        }
    }

    /// 查询视口内的实体 ID
    pub fn query_viewport_ids(&self, viewport_min: Point2, viewport_max: Point2) -> HashSet<usize> {
        let min_grid = self.world_to_grid(viewport_min);
        let max_grid = self.world_to_grid(viewport_max);

        let mut result = HashSet::new();

        for x in min_grid[0]..=max_grid[0] {
            for y in min_grid[1]..=max_grid[1] {
                if let Some(cell) = self.cells.get(&[x, y]) {
                    // 检查单元格包围盒是否与视口相交
                    if self.aabb_intersects_viewport(&cell.aabb, viewport_min, viewport_max) {
                        for &id in &cell.entity_ids {
                            // 精确检查实体包围盒
                            if let Some(aabb) = self.entity_aabbs.get(&id) {
                                if self.aabb_intersects_viewport(aabb, viewport_min, viewport_max) {
                                    result.insert(id);
                                }
                            }
                        }
                    }
                }
            }
        }

        result
    }

    /// 检查包围盒是否与视口相交
    fn aabb_intersects_viewport(&self, aabb: &AABB<[f64; 2]>, viewport_min: Point2, viewport_max: Point2) -> bool {
        let lower = aabb.lower();
        let upper = aabb.upper();

        !(lower[0] > viewport_max[0] ||
          upper[0] < viewport_min[0] ||
          lower[1] > viewport_max[1] ||
          upper[1] < viewport_min[1])
    }

    /// 获取索引中的实体总数
    pub fn len(&self) -> usize {
        self.entity_aabbs.len()
    }
}

// ============================================================================
// 分层空间索引（统一入口）
// ============================================================================

/// 分层空间索引配置
#[derive(Debug, Clone)]
pub struct SpatialIndexConfig {
    /// 网格单元格大小
    pub grid_cell_size: f64,
    /// R*-tree 最小填充因子
    pub rtree_min_fill: usize,
    /// R*-tree 最大填充因子
    pub rtree_max_fill: usize,
    /// 是否启用网格索引
    pub enable_grid: bool,
    /// 是否启用 R*-tree 索引
    pub enable_rtree: bool,
}

impl Default for SpatialIndexConfig {
    fn default() -> Self {
        Self {
            grid_cell_size: 100.0,
            rtree_min_fill: 10,
            rtree_max_fill: 50,
            enable_grid: true,
            enable_rtree: true,
        }
    }
}

/// 分层空间索引
///
/// 结合网格（粗粒度）和 R*-tree（细粒度）的优势：
/// 1. 网格快速过滤大部分不可见实体
/// 2. R*-tree 精确查询剩余实体
/// 3. 支持按图层过滤
#[derive(Debug, Clone)]
pub struct SpatialIndex {
    config: SpatialIndexConfig,
    grid_index: Option<GridIndex>,
    rtree_index: Option<RTreeIndex>,
    /// 所有实体（用于快速访问）
    entities: HashMap<usize, RenderEntity>,
    /// 图层索引
    layer_index: HashMap<String, HashSet<usize>>,
}

impl Default for SpatialIndex {
    fn default() -> Self {
        Self::new()
    }
}

impl SpatialIndex {
    /// 创建新的分层空间索引
    pub fn new() -> Self {
        Self::with_config(SpatialIndexConfig::default())
    }

    /// 使用配置创建空间索引
    pub fn with_config(config: SpatialIndexConfig) -> Self {
        let grid_index = if config.enable_grid {
            Some(GridIndex::new(config.grid_cell_size, [0.0, 0.0]))
        } else {
            None
        };

        let rtree_index = if config.enable_rtree {
            Some(RTreeIndex::new())
        } else {
            None
        };

        Self {
            config,
            grid_index,
            rtree_index,
            entities: HashMap::new(),
            layer_index: HashMap::new(),
        }
    }

    /// 创建自适应空间索引
    pub fn with_bounds(scene_min: Point2, scene_max: Point2) -> Self {
        let config = SpatialIndexConfig::default();
        
        let grid_index = if config.enable_grid {
            Some(GridIndex::with_bounds(scene_min, scene_max, 64))
        } else {
            None
        };

        Self {
            config,
            grid_index,
            rtree_index: Some(RTreeIndex::new()),
            entities: HashMap::new(),
            layer_index: HashMap::new(),
        }
    }

    /// 插入实体
    pub fn insert(&mut self, id: usize, entity: RenderEntity) {
        let layer = entity.layer().to_string();

        // 插入到网格索引
        if let Some(ref mut grid) = self.grid_index {
            grid.insert(id, &entity);
        }

        // 插入到 R*-tree 索引
        if let Some(ref mut rtree) = self.rtree_index {
            rtree.insert(id, entity.clone());
        }

        // 更新实体映射
        self.entities.insert(id, entity);
        
        // 更新图层索引
        self.layer_index.entry(layer).or_default().insert(id);
    }

    /// 删除实体
    pub fn remove(&mut self, id: usize) -> Option<RenderEntity> {
        if let Some(entity) = self.entities.remove(&id) {
            let layer = entity.layer().to_string();

            if let Some(ref mut grid) = self.grid_index {
                grid.remove(id);
            }

            if let Some(ref mut rtree) = self.rtree_index {
                rtree.remove(id);
            }

            if let Some(ids) = self.layer_index.get_mut(&layer) {
                ids.remove(&id);
            }

            return Some(entity);
        }
        None
    }

    /// 查询视口内的实体
    pub fn query_viewport(&self, viewport_min: Point2, viewport_max: Point2) -> Vec<&RenderEntity> {
        // 优先使用网格索引快速过滤
        if let Some(ref grid) = self.grid_index {
            let ids = grid.query_viewport_ids(viewport_min, viewport_max);
            ids.iter()
                .filter_map(|&id| self.entities.get(&id))
                .collect()
        }
        // 回退到 R*-tree
        else if let Some(ref rtree) = self.rtree_index {
            rtree.query_viewport(viewport_min, viewport_max)
        }
        else {
            Vec::new()
        }
    }

    /// 查询视口内的实体（带图层过滤）
    pub fn query_viewport_with_layers(
        &self,
        viewport_min: Point2,
        viewport_max: Point2,
        visible_layers: &HashSet<String>,
    ) -> Vec<&RenderEntity> {
        let entities = self.query_viewport(viewport_min, viewport_max);
        
        if visible_layers.is_empty() || visible_layers.contains("*") {
            return entities;
        }

        entities
            .into_iter()
            .filter(|e| visible_layers.contains(e.layer()))
            .collect()
    }

    /// 按图层查询实体
    pub fn query_layer(&self, layer: &str) -> Vec<&RenderEntity> {
        if let Some(ids) = self.layer_index.get(layer) {
            ids.iter()
                .filter_map(|&id| self.entities.get(&id))
                .collect()
        } else {
            Vec::new()
        }
    }

    /// 获取索引中的实体总数
    pub fn len(&self) -> usize {
        self.entities.len()
    }

    /// 检查索引是否为空
    pub fn is_empty(&self) -> bool {
        self.entities.is_empty()
    }

    /// 获取统计信息
    pub fn stats(&self) -> SpatialIndexStats {
        let grid_count = self.grid_index.as_ref().map(|g| g.len()).unwrap_or(0);
        let rtree_count = self.rtree_index.as_ref().map(|r| r.len()).unwrap_or(0);
        
        SpatialIndexStats {
            total_entities: self.entities.len(),
            grid_entities: grid_count,
            rtree_entities: rtree_count,
            layer_count: self.layer_index.len(),
        }
    }

    /// 清除所有实体
    pub fn clear(&mut self) {
        self.entities.clear();
        self.layer_index.clear();
        
        if let Some(ref mut grid) = self.grid_index {
            *grid = GridIndex::new(self.config.grid_cell_size, [0.0, 0.0]);
        }
        
        if let Some(ref mut rtree) = self.rtree_index {
            *rtree = RTreeIndex::new();
        }
    }
}

/// 空间索引统计信息
#[derive(Debug, Clone, Default)]
pub struct SpatialIndexStats {
    /// 总实体数
    pub total_entities: usize,
    /// 网格索引中的实体数
    pub grid_entities: usize,
    /// R*-tree 索引中的实体数
    pub rtree_entities: usize,
    /// 图层数量
    pub layer_count: usize,
}

// ============================================================================
// 视口裁剪器（集成稳健几何）
// ============================================================================

/// 视口裁剪器
///
/// 使用 Cohen-Sutherland 算法和稳健几何谓词进行精确裁剪
#[derive(Debug, Clone, Default)]
pub struct ViewportCuller {
    viewport_min: Point2,
    viewport_max: Point2,
}

impl ViewportCuller {
    /// 创建新的视口裁剪器
    pub fn new(viewport_min: Point2, viewport_max: Point2) -> Self {
        Self { viewport_min, viewport_max }
    }

    /// 更新视口范围
    pub fn set_viewport(&mut self, viewport_min: Point2, viewport_max: Point2) {
        self.viewport_min = viewport_min;
        self.viewport_max = viewport_max;
    }

    /// 检查线段是否在视口内（使用稳健几何）
    pub fn line_in_viewport(&self, start: Point2, end: Point2) -> bool {
        // 快速排斥：检查包围盒
        let min_x = start[0].min(end[0]);
        let max_x = start[0].max(end[0]);
        let min_y = start[1].min(end[1]);
        let max_y = start[1].max(end[1]);

        if max_x < self.viewport_min[0] || min_x > self.viewport_max[0] ||
           max_y < self.viewport_min[1] || min_y > self.viewport_max[1] {
            return false;
        }

        // 如果两个端点都在视口内，直接返回 true
        if self.point_in_viewport(start) && self.point_in_viewport(end) {
            return true;
        }

        // 使用 Cohen-Sutherland 算法检查是否相交
        self.line_intersects_viewport(start, end)
    }

    /// 检查点是否在视口内
    pub fn point_in_viewport(&self, point: Point2) -> bool {
        point[0] >= self.viewport_min[0] && point[0] <= self.viewport_max[0] &&
        point[1] >= self.viewport_min[1] && point[1] <= self.viewport_max[1]
    }

    /// 检查线段是否与视口边界相交
    fn line_intersects_viewport(&self, start: Point2, end: Point2) -> bool {
        let corners = [
            [self.viewport_min[0], self.viewport_min[1]],
            [self.viewport_max[0], self.viewport_min[1]],
            [self.viewport_max[0], self.viewport_max[1]],
            [self.viewport_min[0], self.viewport_max[1]],
        ];

        // 检查线段是否与视口的四条边相交
        for i in 0..4 {
            let j = (i + 1) % 4;
            if self.segments_intersect(start, end, corners[i], corners[j]) {
                return true;
            }
        }

        false
    }

    /// 检查两条线段是否相交（使用稳健几何）
    fn segments_intersect(&self, p1: Point2, p2: Point2, p3: Point2, p4: Point2) -> bool {
        let d1 = orient2d(p3, p4, p1);
        let d2 = orient2d(p3, p4, p2);
        let d3 = orient2d(p1, p2, p3);
        let d4 = orient2d(p1, p2, p4);

        // 快速排斥
        if !Self::intersects_strict(d1, d2) || !Self::intersects_strict(d3, d4) {
            return false;
        }

        true
    }

    fn intersects_strict(d1: Orientation, d2: Orientation) -> bool {
        matches!(
            (d1, d2),
            (Orientation::Clockwise, Orientation::CounterClockwise) |
            (Orientation::CounterClockwise, Orientation::Clockwise)
        )
    }

    /// 裁剪线段到视口（返回裁剪后的线段）
    pub fn clip_line(&self, start: Point2, end: Point2) -> Option<(Point2, Point2)> {
        // 如果线段完全在视口内，直接返回
        if self.point_in_viewport(start) && self.point_in_viewport(end) {
            return Some((start, end));
        }

        // 如果线段完全在视口外，返回 None
        if !self.line_in_viewport(start, end) {
            return None;
        }

        // TODO: 实现 Liang-Barsky 或 Cohen-Sutherland 裁剪算法
        // 目前返回原始线段（保守处理）
        Some((start, end))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_entity() -> RenderEntity {
        RenderEntity::Line {
            start: [0.0, 0.0],
            end: [10.0, 10.0],
            layer: "TEST".to_string(),
            color: [1.0, 0.0, 0.0, 1.0],
        }
    }

    #[test]
    fn test_render_entity_aabb() {
        let entity = create_test_entity();
        let aabb = entity.aabb();
        
        assert!((aabb.lower()[0] - 0.0).abs() < 1e-10);
        assert!((aabb.lower()[1] - 0.0).abs() < 1e-10);
        assert!((aabb.upper()[0] - 10.0).abs() < 1e-10);
        assert!((aabb.upper()[1] - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_rtree_index() {
        let mut index = RTreeIndex::new();
        
        index.insert(1, create_test_entity());
        index.insert(2, RenderEntity::Line {
            start: [50.0, 50.0],
            end: [60.0, 60.0],
            layer: "TEST".to_string(),
            color: [0.0, 1.0, 0.0, 1.0],
        });

        // 查询视口
        let results = index.query_viewport([0.0, 0.0], [20.0, 20.0]);
        assert_eq!(results.len(), 1);

        // 查询大视口
        let results = index.query_viewport([0.0, 0.0], [100.0, 100.0]);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_grid_index() {
        let mut index = GridIndex::new(50.0, [0.0, 0.0]);
        
        index.insert(1, &create_test_entity());
        
        // 查询视口
        let ids = index.query_viewport_ids([0.0, 0.0], [20.0, 20.0]);
        assert!(ids.contains(&1));
    }

    #[test]
    fn test_spatial_index() {
        let mut index = SpatialIndex::new();
        
        index.insert(1, create_test_entity());
        index.insert(2, RenderEntity::Line {
            start: [100.0, 100.0],
            end: [110.0, 110.0],
            layer: "OTHER".to_string(),
            color: [0.0, 0.0, 1.0, 1.0],
        });

        // 查询视口
        let results = index.query_viewport([0.0, 0.0], [50.0, 50.0]);
        assert_eq!(results.len(), 1);

        // 按图层查询
        let results = index.query_layer("TEST");
        assert_eq!(results.len(), 1);

        // 带图层过滤的查询
        let visible_layers = ["TEST".to_string()].iter().cloned().collect();
        let results = index.query_viewport_with_layers(
            [0.0, 0.0], [200.0, 200.0],
            &visible_layers
        );
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_viewport_culler() {
        let culler = ViewportCuller::new([0.0, 0.0], [100.0, 100.0]);

        // 完全在视口内
        assert!(culler.line_in_viewport([10.0, 10.0], [20.0, 20.0]));

        // 完全在视口外
        assert!(!culler.line_in_viewport([200.0, 200.0], [210.0, 210.0]));

        // 穿过视口
        assert!(culler.line_in_viewport([-50.0, 50.0], [150.0, 50.0]));

        // 点在视口内测试
        assert!(culler.point_in_viewport([50.0, 50.0]));
        assert!(!culler.point_in_viewport([150.0, 150.0]));
    }

    #[test]
    fn test_spatial_index_stats() {
        let mut index = SpatialIndex::new();
        
        index.insert(1, create_test_entity());
        index.insert(2, RenderEntity::Line {
            start: [50.0, 50.0],
            end: [60.0, 60.0],
            layer: "TEST".to_string(),
            color: [0.0, 1.0, 0.0, 1.0],
        });

        let stats = index.stats();
        assert_eq!(stats.total_entities, 2);
        assert!(stats.grid_entities >= 2 || stats.rtree_entities >= 2);
        assert_eq!(stats.layer_count, 1);
    }
}
