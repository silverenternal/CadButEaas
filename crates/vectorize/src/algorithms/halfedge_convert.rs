//! Halfedge 网格转换器：从矢量化结果构建 Halfedge 图
//!
//! 将 vectorize 的输出（strokes/loops）转换为 topo crate 的 HalfedgeGraph，
//! 无缝对接现有拓扑分析功能。

use super::topology_stitch::{Loop, TopologyStitcher};
use super::tracing::Stroke;
use common_types::geometry::Point2;

/// Halfedge 转换结果
#[derive(Debug, Clone)]
pub struct HalfedgeConversionResult {
    /// 外轮廓索引列表
    pub outer_loop_indices: Vec<usize>,
    /// 所有环的点集合
    pub all_loop_points: Vec<Vec<Point2>>,
    /// 转换统计
    pub stats: ConversionStats,
}

/// 转换统计信息
#[derive(Debug, Clone, Default)]
pub struct ConversionStats {
    /// 输入笔画数
    pub input_strokes: usize,
    /// 提取的环总数
    pub total_loops: usize,
    /// 外轮廓数量
    pub outer_loops: usize,
    /// 孔洞数量
    pub hole_loops: usize,
    /// 总顶点数
    pub total_vertices: usize,
}

/// Halfedge 转换器
#[derive(Debug, Clone)]
pub struct HalfedgeConverter {
    /// 拓扑缝合器
    stitcher: TopologyStitcher,
    /// 是否合并相邻点（去重）
    pub deduplicate_points: bool,
    /// 合并点的容差
    pub dedup_tolerance: f64,
}

impl Default for HalfedgeConverter {
    fn default() -> Self {
        Self {
            stitcher: TopologyStitcher::new(),
            deduplicate_points: true,
            dedup_tolerance: 1e-6,
        }
    }
}

impl HalfedgeConverter {
    /// 创建新的转换器
    pub fn new() -> Self {
        Self::default()
    }

    /// 设置吸附容差
    pub fn with_snap_tolerance(mut self, tolerance: f64) -> Self {
        self.stitcher.snap_tolerance = tolerance;
        self.stitcher.auto_tolerance = false;
        self
    }

    /// 启用/禁用自动容差计算
    pub fn with_auto_tolerance(mut self, auto: bool) -> Self {
        self.stitcher.auto_tolerance = auto;
        self
    }

    /// 启用/禁用点去重
    pub fn with_deduplication(mut self, enabled: bool) -> Self {
        self.deduplicate_points = enabled;
        self
    }

    /// 去重相邻重复点（减少冗余顶点）
    fn deduplicate_loop(&self, points: &[Point2]) -> Vec<Point2> {
        if points.len() < 2 {
            return points.to_vec();
        }

        let mut result = Vec::with_capacity(points.len());
        result.push(points[0]);

        for &pt in &points[1..] {
            let last = result[result.len() - 1];
            let dx = pt[0] - last[0];
            let dy = pt[1] - last[1];
            let dist_sq = dx * dx + dy * dy;
            if dist_sq > self.dedup_tolerance * self.dedup_tolerance {
                result.push(pt);
            }
        }

        // 检查首尾是否相同（闭环）
        if result.len() >= 2 {
            let first = result[0];
            let last = result[result.len() - 1];
            let dx = first[0] - last[0];
            let dy = first[1] - last[1];
            let dist_sq = dx * dx + dy * dy;
            if dist_sq <= self.dedup_tolerance * self.dedup_tolerance {
                result.pop();
            }
        }

        result
    }

    /// 从 strokes 构建 HalfedgeGraph
    /// 这是两步流程：先拓扑缝合提取环，再构建 Halfedge 图
    pub fn convert_strokes(&mut self, strokes: &[Stroke]) -> HalfedgeConversionResult {
        // Step 1: 拓扑缝合提取环
        let loops = self.stitcher.stitch(strokes);

        self.convert_loops(&loops, strokes.len())
    }

    /// 从已有的 Loop 集合构建转换结果
    pub fn convert_loops(
        &self,
        loops: &[Loop],
        input_stroke_count: usize,
    ) -> HalfedgeConversionResult {
        // 收集所有环的点
        let mut all_loop_points = Vec::with_capacity(loops.len());
        let mut outer_indices = Vec::new();
        let mut total_vertices = 0;

        for (i, lp) in loops.iter().enumerate() {
            let points = if self.deduplicate_points {
                self.deduplicate_loop(&lp.points)
            } else {
                lp.points.clone()
            };

            total_vertices += points.len();
            all_loop_points.push(points);

            if lp.is_outer && lp.parent.is_none() {
                outer_indices.push(i);
            }
        }

        let hole_count = loops
            .iter()
            .filter(|l| !l.is_outer || l.parent.is_some())
            .count();

        HalfedgeConversionResult {
            outer_loop_indices: outer_indices.clone(),
            all_loop_points,
            stats: ConversionStats {
                input_strokes: input_stroke_count,
                total_loops: loops.len(),
                outer_loops: outer_indices.len(),
                hole_loops: hole_count,
                total_vertices,
            },
        }
    }

    /// 从 strokes 直接生成可传给 topo crate 的环数据
    /// 这是便捷方法，直接返回 topo crate 需要的格式
    pub fn prepare_for_topo(&mut self, strokes: &[Stroke]) -> Vec<Vec<Point2>> {
        let result = self.convert_strokes(strokes);
        result.all_loop_points
    }

    /// 获取外轮廓的点集合（用于 topo crate 处理）
    pub fn get_outer_loops<'a>(&self, result: &'a HalfedgeConversionResult) -> Vec<&'a [Point2]> {
        result
            .outer_loop_indices
            .iter()
            .map(|&i| result.all_loop_points[i].as_slice())
            .collect()
    }

    /// 获取指定外轮廓的孔洞点集合
    pub fn get_holes_for_outer<'a>(
        &self,
        loops: &'a [Loop],
        result: &'a HalfedgeConversionResult,
        outer_idx: usize,
    ) -> Vec<&'a [Point2]> {
        if outer_idx >= loops.len() {
            return Vec::new();
        }

        loops[outer_idx]
            .children
            .iter()
            .map(|&child_idx| result.all_loop_points[child_idx].as_slice())
            .collect()
    }
}

/// 便捷函数：直接将 strokes 转换为 topo crate 可用的环格式
pub fn strokes_to_topo_loops(strokes: &[Stroke], snap_tolerance: Option<f64>) -> Vec<Vec<Point2>> {
    let mut converter = HalfedgeConverter::new();
    if let Some(tol) = snap_tolerance {
        converter = converter.with_snap_tolerance(tol);
    }
    converter.prepare_for_topo(strokes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithms::SubpixelPoint;

    fn create_square_strokes() -> Vec<Stroke> {
        // 创建一个正方形的四条边
        vec![
            Stroke {
                start: SubpixelPoint { x: 0.0, y: 0.0 },
                end: SubpixelPoint { x: 10.0, y: 0.0 },
                points: Vec::new(),
                length: 10.0,
                is_closed: false,
            },
            Stroke {
                start: SubpixelPoint { x: 10.0, y: 0.0 },
                end: SubpixelPoint { x: 10.0, y: 10.0 },
                points: Vec::new(),
                length: 10.0,
                is_closed: false,
            },
            Stroke {
                start: SubpixelPoint { x: 10.0, y: 10.0 },
                end: SubpixelPoint { x: 0.0, y: 10.0 },
                points: Vec::new(),
                length: 10.0,
                is_closed: false,
            },
            Stroke {
                start: SubpixelPoint { x: 0.0, y: 10.0 },
                end: SubpixelPoint { x: 0.0, y: 0.0 },
                points: Vec::new(),
                length: 10.0,
                is_closed: false,
            },
        ]
    }

    #[test]
    fn test_converter_creation() {
        let converter = HalfedgeConverter::new();
        assert!(converter.deduplicate_points);
        assert!((converter.dedup_tolerance - 1e-6).abs() < 1e-10);
    }

    #[test]
    fn test_with_snap_tolerance() {
        let converter = HalfedgeConverter::new().with_snap_tolerance(3.5);
        assert!((converter.stitcher.snap_tolerance - 3.5).abs() < 1e-10);
        assert!(!converter.stitcher.auto_tolerance);
    }

    #[test]
    fn test_deduplicate_loop() {
        let converter = HalfedgeConverter::new();
        let points = vec![
            [0.0, 0.0],
            [0.0000001, 0.0], // 几乎相同的点
            [10.0, 0.0],
            [10.0, 10.0],
            [0.0, 10.0],
        ];
        let deduped = converter.deduplicate_loop(&points);
        assert_eq!(deduped.len(), 4); // 去掉了一个重复点
    }

    #[test]
    fn test_convert_strokes_basic() {
        let strokes = create_square_strokes();
        let mut converter = HalfedgeConverter::new();
        let result = converter.convert_strokes(&strokes);

        assert_eq!(result.stats.input_strokes, 4);
        // 至少应该有一个环（外轮廓）
        assert!(result.stats.total_loops >= 1);
        assert!(!result.all_loop_points.is_empty());
    }

    #[test]
    fn test_prepare_for_topo() {
        let strokes = create_square_strokes();
        let loops = strokes_to_topo_loops(&strokes, Some(2.0));
        // 返回的是 topo crate 可以直接使用的格式
        assert!(!loops.is_empty());
    }

    #[test]
    fn test_get_outer_loops() {
        let strokes = create_square_strokes();
        let mut converter = HalfedgeConverter::new();
        let result = converter.convert_strokes(&strokes);
        let outer = converter.get_outer_loops(&result);
        assert!(!outer.is_empty());
    }

    #[test]
    fn test_conversion_stats() {
        let strokes = create_square_strokes();
        let mut converter = HalfedgeConverter::new();
        let result = converter.convert_strokes(&strokes);

        assert_eq!(result.stats.input_strokes, 4);
        assert!(result.stats.total_vertices > 0);
        assert!(result.stats.outer_loops >= 1);
    }

    #[test]
    fn test_without_deduplication() {
        let converter = HalfedgeConverter::new().with_deduplication(false);
        assert!(!converter.deduplicate_points);
    }

    #[test]
    fn test_with_auto_tolerance() {
        let converter = HalfedgeConverter::new().with_auto_tolerance(true);
        assert!(converter.stitcher.auto_tolerance);
    }
}
