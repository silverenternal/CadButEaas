//! 空间索引（P1-2 修复：R*-tree 视口裁剪优化）
//!
//! 使用 R*-tree 空间索引加速视口裁剪查询，避免 O(n) 遍历所有边

use rstar::RTree;
use rstar::AABB;
use egui::Rect;

/// 空间索引项
#[derive(Debug, Clone)]
pub struct SpatialItem {
    /// 边 ID
    pub edge_id: usize,
    /// 边的包围盒
    pub aabb: AABB<[f64; 2]>,
    /// 起点
    pub start: [f64; 2],
    /// 终点
    pub end: [f64; 2],
}

impl rstar::RTreeObject for SpatialItem {
    type Envelope = AABB<[f64; 2]>;

    fn envelope(&self) -> Self::Envelope {
        self.aabb
    }
}

/// 空间索引（R*-tree）
pub struct SpatialIndex {
    tree: RTree<SpatialItem>,
}

impl SpatialIndex {
    /// 创建新的空间索引
    pub fn new() -> Self {
        Self {
            tree: RTree::new(),
        }
    }

    /// 从边列表构建空间索引
    pub fn from_edges(edges: &[(usize, [f64; 2], [f64; 2])]) -> Self {
        let items: Vec<SpatialItem> = edges
            .iter()
            .map(|(id, start, end)| {
                let aabb = AABB::from_corners(*start, *end);
                SpatialItem {
                    edge_id: *id,
                    aabb,
                    start: *start,
                    end: *end,
                }
            })
            .collect();

        Self {
            tree: RTree::bulk_load(items),
        }
    }

    /// 查询视口内的边 ID
    pub fn query_in_viewport(&self, viewport: Rect) -> Vec<usize> {
        let query_box = AABB::from_corners(
            [viewport.left() as f64, viewport.bottom() as f64],
            [viewport.right() as f64, viewport.top() as f64],
        );

        self.tree
            .locate_in_envelope(&query_box)
            .map(|item| item.edge_id)
            .collect()
    }

    /// 查询视口内的边（带边距，避免边缘裁剪）
    pub fn query_in_viewport_with_margin(
        &self,
        viewport: Rect,
        margin: f64,
    ) -> Vec<usize> {
        let expanded_viewport = Rect::from_min_max(
            egui::pos2(
                (viewport.left() as f64 - margin) as f32,
                (viewport.top() as f64 - margin) as f32,
            ),
            egui::pos2(
                (viewport.right() as f64 + margin) as f32,
                (viewport.bottom() as f64 + margin) as f32,
            ),
        );

        self.query_in_viewport(expanded_viewport)
    }

    /// 获取索引中的项目数量
    pub fn len(&self) -> usize {
        self.tree.len()
    }

    /// 检查索引是否为空
    pub fn is_empty(&self) -> bool {
        self.tree.is_empty()
    }
}

impl Default for SpatialIndex {
    fn default() -> Self {
        Self::new()
    }
}

/// P1-2 新增：视口裁剪优化器
pub struct ViewportCuller {
    /// 空间索引
    spatial_index: Option<SpatialIndex>,
    /// 边数据缓存（用于快速查找）
    edge_cache: std::collections::HashMap<usize, ([f64; 2], [f64; 2])>,
}

impl ViewportCuller {
    /// 创建新的视口裁剪器
    pub fn new() -> Self {
        Self {
            spatial_index: None,
            edge_cache: std::collections::HashMap::new(),
        }
    }

    /// 更新空间索引（当场景变化时调用）
    pub fn update_index(&mut self, edges: &[(usize, [f64; 2], [f64; 2])]) {
        self.spatial_index = Some(SpatialIndex::from_edges(edges));
        self.edge_cache = edges
            .iter()
            .map(|(id, start, end)| (*id, (*start, *end)))
            .collect();
    }

    /// 获取视口内的边
    pub fn get_visible_edges(
        &self,
        viewport: Rect,
        margin: f64,
    ) -> Vec<(usize, [f64; 2], [f64; 2])> {
        if let Some(index) = &self.spatial_index {
            let visible_ids = index.query_in_viewport_with_margin(viewport, margin);
            visible_ids
                .into_iter()
                .filter_map(|id| {
                    self.edge_cache.get(&id).map(|&(start, end)| (id, start, end))
                })
                .collect()
        } else {
            // 没有索引，返回空
            Vec::new()
        }
    }

    /// 清除索引
    pub fn clear(&mut self) {
        self.spatial_index = None;
        self.edge_cache.clear();
    }
}

impl Default for ViewportCuller {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spatial_index_creation() {
        let edges = vec![
            (0, [0.0, 0.0], [10.0, 10.0]),
            (1, [5.0, 5.0], [15.0, 15.0]),
            (2, [100.0, 100.0], [110.0, 110.0]),
        ];

        let index = SpatialIndex::from_edges(&edges);
        assert_eq!(index.len(), 3);
    }

    #[test]
    fn test_viewport_query() {
        let edges = vec![
            (0, [0.0, 0.0], [10.0, 10.0]),
            (1, [5.0, 5.0], [15.0, 15.0]),
            (2, [100.0, 100.0], [110.0, 110.0]),
        ];

        let index = SpatialIndex::from_edges(&edges);
        let viewport = Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(20.0, 20.0));

        let visible = index.query_in_viewport(viewport);
        assert!(visible.contains(&0));
        assert!(visible.contains(&1));
        assert!(!visible.contains(&2));
    }

    #[test]
    fn test_viewport_culler() {
        let mut culler = ViewportCuller::new();
        let edges = vec![
            (0, [0.0, 0.0], [10.0, 10.0]),
            (1, [5.0, 5.0], [15.0, 15.0]),
            (2, [100.0, 100.0], [110.0, 110.0]),
        ];

        culler.update_index(&edges);
        let viewport = Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(20.0, 20.0));
        let visible = culler.get_visible_edges(viewport, 0.0);

        assert_eq!(visible.len(), 2);
    }
}
