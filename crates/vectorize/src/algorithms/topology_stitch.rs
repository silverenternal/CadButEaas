//! 拓扑缝合算法：从 strokes 到闭合环
//!
//! 将矢量化的笔画（strokes）缝合为拓扑正确的闭合区域：
//! - 端点自适应吸附
//! - 环的正确遍历（顺时针/逆时针）
//! - 孔洞嵌套关系识别
//! - 嵌套层级计算

use super::tracing::Stroke;
use common_types::geometry::Point2;
use std::collections::HashSet;

/// 缝合后的环
#[derive(Debug, Clone)]
pub struct Loop {
    /// 环的顶点序列
    pub points: Vec<Point2>,
    /// 是否为外轮廓（逆时针）
    pub is_outer: bool,
    /// 环的面积（有符号）
    pub signed_area: f64,
    /// 嵌套层级（0 为最外层）
    pub nesting_level: usize,
    /// 父环 ID（如果是孔洞）
    pub parent: Option<usize>,
    /// 子环 ID 列表（孔洞）
    pub children: Vec<usize>,
}

impl Loop {
    /// 计算多边形有符号面积（ shoelace 公式）
    /// 逆时针为正，顺时针为负
    pub fn compute_signed_area(points: &[Point2]) -> f64 {
        if points.len() < 3 {
            return 0.0;
        }
        let mut area = 0.0;
        let n = points.len();
        for i in 0..n {
            let j = (i + 1) % n;
            area += points[i][0] * points[j][1];
            area -= points[j][0] * points[i][1];
        }
        area / 2.0
    }

    /// 检查点是否在多边形内（射线法）
    pub fn contains_point(&self, point: Point2) -> bool {
        if self.points.len() < 3 {
            return false;
        }
        let mut inside = false;
        let n = self.points.len();
        for i in 0..n {
            let j = (i + 1) % n;
            let a = self.points[i];
            let b = self.points[j];

            // 检查点是否在边的 y 范围内
            if (a[1] > point[1]) != (b[1] > point[1]) {
                // 计算交点的 x 坐标
                let t = (point[1] - a[1]) / (b[1] - a[1]);
                let x_intersect = a[0] + t * (b[0] - a[0]);
                if point[0] < x_intersect {
                    inside = !inside;
                }
            }
        }
        inside
    }

    /// 检查另一个环是否在本环内
    pub fn contains_loop(&self, other: &Loop) -> bool {
        if other.points.is_empty() {
            return false;
        }
        // 取另一个环的第一个点做包含测试
        self.contains_point(other.points[0])
    }
}

/// 端点顶点
#[derive(Debug, Clone)]
struct Vertex {
    /// 位置
    pub position: Point2,
    /// 连接的边索引
    pub edges: Vec<usize>,
}

/// 有向边（从 start 到 end）
#[derive(Debug, Clone)]
struct DirectedEdge {
    /// 起点索引
    pub start: usize,
    /// 终点索引
    pub end: usize,
    /// 原始 stroke 索引
    pub _stroke_idx: usize,
    /// 是否反向（相对于原始 stroke）
    pub _reversed: bool,
}

/// 拓扑缝合器
#[derive(Debug, Clone)]
pub struct TopologyStitcher {
    /// 吸附容差（像素）
    pub snap_tolerance: f64,
    /// 是否自动计算容差
    pub auto_tolerance: bool,
    /// 最小环边数
    pub min_loop_edges: usize,
}

impl Default for TopologyStitcher {
    fn default() -> Self {
        Self {
            snap_tolerance: 2.0,
            auto_tolerance: true,
            min_loop_edges: 3,
        }
    }
}

impl TopologyStitcher {
    /// 创建新的缝合器
    pub fn new() -> Self {
        Self::default()
    }

    /// 根据笔画平均长度自适应计算容差
    pub fn compute_auto_tolerance(&self, strokes: &[Stroke]) -> f64 {
        if strokes.is_empty() {
            return 2.0;
        }
        let total_length: f64 = strokes.iter().map(|s| s.length).sum();
        let avg_length = total_length / strokes.len() as f64;
        // 容差为平均长度的 2%，最小 1 像素，最大 5 像素
        (avg_length * 0.02).clamp(1.0, 5.0)
    }

    /// 找出距离给定点最近的顶点索引
    fn find_nearest_vertex(&self, vertices: &[Vertex], point: Point2) -> Option<usize> {
        vertices
            .iter()
            .enumerate()
            .map(|(i, v)| {
                let dx = v.position[0] - point[0];
                let dy = v.position[1] - point[1];
                let dist = (dx * dx + dy * dy).sqrt();
                (i, dist)
            })
            .filter(|&(_, d)| d <= self.snap_tolerance)
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .map(|(i, _)| i)
    }

    /// 添加或吸附顶点
    fn add_or_snap_vertex(&self, vertices: &mut Vec<Vertex>, point: Point2) -> usize {
        if let Some(idx) = self.find_nearest_vertex(vertices, point) {
            idx
        } else {
            vertices.push(Vertex {
                position: point,
                edges: Vec::new(),
            });
            vertices.len() - 1
        }
    }

    /// 从 strokes 构建有向边图
    fn build_edge_graph(&self, strokes: &[Stroke]) -> (Vec<Vertex>, Vec<DirectedEdge>) {
        let mut vertices = Vec::new();
        let mut edges = Vec::new();

        for (stroke_idx, stroke) in strokes.iter().enumerate() {
            // 将起点和终点吸附到现有顶点
            let start = [stroke.start.x, stroke.start.y];
            let end = [stroke.end.x, stroke.end.y];

            let v_start = self.add_or_snap_vertex(&mut vertices, start);
            let v_end = self.add_or_snap_vertex(&mut vertices, end);

            // 添加正反两条有向边
            edges.push(DirectedEdge {
                start: v_start,
                end: v_end,
                _stroke_idx: stroke_idx,
                _reversed: false,
            });
            edges.push(DirectedEdge {
                start: v_end,
                end: v_start,
                _stroke_idx: stroke_idx,
                _reversed: true,
            });

            // 记录边连接关系
            let e_idx = edges.len() - 2;
            vertices[v_start].edges.push(e_idx);
            vertices[v_end].edges.push(e_idx + 1);
        }

        (vertices, edges)
    }

    /// 提取所有闭合环
    fn extract_loops(
        &self,
        vertices: &[Vertex],
        edges: &[DirectedEdge],
        _strokes: &[Stroke],
    ) -> Vec<Loop> {
        let mut visited = HashSet::new();
        let mut loops = Vec::new();

        for start_edge in 0..edges.len() {
            if visited.contains(&start_edge) {
                continue;
            }

            let mut current_edge = start_edge;
            let mut loop_points = Vec::new();
            let mut loop_edges = Vec::new();

            // 沿着 next 边遍历（这里使用简单的转角规则）
            loop {
                if visited.contains(&current_edge) {
                    break;
                }
                visited.insert(current_edge);
                loop_edges.push(current_edge);

                let edge = &edges[current_edge];
                loop_points.push(vertices[edge.start].position);

                // 找到下一条边：从 end 顶点出发的除了 twin 之外的边
                let twin = if current_edge % 2 == 0 {
                    current_edge + 1
                } else {
                    current_edge - 1
                };

                let from_vertex = edge.end;
                let outgoing_edges: Vec<usize> = vertices[from_vertex]
                    .edges
                    .iter()
                    .filter(|&&e| e != twin)
                    .copied()
                    .collect();

                if outgoing_edges.is_empty() {
                    break; // 死路，不是闭环
                }

                // 简单策略：取第一条边
                current_edge = outgoing_edges[0];
            }

            // 检查是否形成了有效闭环
            if loop_points.len() >= self.min_loop_edges {
                let first = edges[loop_edges[0]].start;
                let last = edges[loop_edges[loop_edges.len() - 1]].end;
                if first == last {
                    // 是闭环
                    let signed_area = Loop::compute_signed_area(&loop_points);
                    loops.push(Loop {
                        points: loop_points,
                        is_outer: signed_area > 0.0,
                        signed_area,
                        nesting_level: 0,
                        parent: None,
                        children: Vec::new(),
                    });
                }
            }
        }

        loops
    }

    /// 计算环的嵌套关系
    fn compute_nesting_hierarchy(&self, loops: &mut [Loop]) {
        if loops.is_empty() {
            return;
        }

        // 按面积绝对值排序（从大到小）
        let mut indices: Vec<usize> = (0..loops.len()).collect();
        indices.sort_by(|&a, &b| {
            loops[b]
                .signed_area
                .abs()
                .partial_cmp(&loops[a].signed_area.abs())
                .unwrap()
        });

        // 为每个环找到父环（包含它的最小外环）
        for i in 1..indices.len() {
            let child_idx = indices[i];
            let mut parent_idx = None;
            let mut parent_area = f64::MAX;

            for &j in &indices[0..i] {
                if loops[j].contains_loop(&loops[child_idx]) {
                    let area = loops[j].signed_area.abs();
                    if area < parent_area {
                        parent_area = area;
                        parent_idx = Some(j);
                    }
                }
            }

            if let Some(p) = parent_idx {
                loops[child_idx].parent = Some(p);
                loops[p].children.push(child_idx);
                loops[child_idx].nesting_level = loops[p].nesting_level + 1;
            }
        }
    }

    /// 执行完整的拓扑缝合流程
    pub fn stitch(&mut self, strokes: &[Stroke]) -> Vec<Loop> {
        // 1. 自动计算容差（如果启用）
        if self.auto_tolerance {
            self.snap_tolerance = self.compute_auto_tolerance(strokes);
        }

        // 2. 构建边图
        let (vertices, edges) = self.build_edge_graph(strokes);

        // 3. 提取闭合环
        let mut loops = self.extract_loops(&vertices, &edges, strokes);

        // 4. 计算嵌套关系
        self.compute_nesting_hierarchy(&mut loops);

        loops
    }

    /// 获取所有外轮廓（层级为 0 且逆时针）
    pub fn outer_loops(&self, loops: &[Loop]) -> Vec<usize> {
        loops
            .iter()
            .enumerate()
            .filter(|(_, l)| l.is_outer && l.parent.is_none())
            .map(|(i, _)| i)
            .collect()
    }

    /// 获取指定环的所有孔洞
    pub fn holes_of_loop(&self, loops: &[Loop], loop_idx: usize) -> Vec<usize> {
        loops[loop_idx].children.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithms::SubpixelPoint;

    fn create_test_stroke(x1: f64, y1: f64, x2: f64, y2: f64) -> Stroke {
        let start = SubpixelPoint { x: x1, y: y1 };
        let end = SubpixelPoint { x: x2, y: y2 };
        let length = start.distance(&end);
        Stroke {
            start,
            end,
            points: Vec::new(),
            length,
            is_closed: false,
        }
    }

    #[test]
    fn test_signed_area_square() {
        // 逆时针正方形（面积 = 100）
        let points = vec![[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]];
        let area = Loop::compute_signed_area(&points);
        assert!((area - 100.0).abs() < 1e-6);
    }

    #[test]
    fn test_signed_area_clockwise() {
        // 顺时针正方形（面积 = -100）
        let points = vec![[0.0, 0.0], [0.0, 10.0], [10.0, 10.0], [10.0, 0.0]];
        let area = Loop::compute_signed_area(&points);
        assert!((area + 100.0).abs() < 1e-6);
    }

    #[test]
    fn test_contains_point_inside() {
        let points = vec![[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]];
        let lp = Loop {
            points,
            is_outer: true,
            signed_area: 100.0,
            nesting_level: 0,
            parent: None,
            children: Vec::new(),
        };
        assert!(lp.contains_point([5.0, 5.0]));
    }

    #[test]
    fn test_contains_point_outside() {
        let points = vec![[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]];
        let lp = Loop {
            points,
            is_outer: true,
            signed_area: 100.0,
            nesting_level: 0,
            parent: None,
            children: Vec::new(),
        };
        assert!(!lp.contains_point([15.0, 5.0]));
    }

    #[test]
    fn test_auto_tolerance_computation() {
        let stitches = [
            create_test_stroke(0.0, 0.0, 100.0, 0.0),     // 长度 100
            create_test_stroke(100.0, 0.0, 100.0, 100.0), // 长度 100
            create_test_stroke(100.0, 100.0, 0.0, 100.0), // 长度 100
            create_test_stroke(0.0, 100.0, 0.0, 0.0),     // 长度 100
        ];
        let stitcher = TopologyStitcher::new();
        let tolerance = stitcher.compute_auto_tolerance(&stitches);
        // 平均长度 100，2% = 2.0，在 [1.0, 5.0] 范围内
        assert!((tolerance - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_stitcher_creation() {
        let stitcher = TopologyStitcher::new();
        assert!(stitcher.auto_tolerance);
        assert_eq!(stitcher.min_loop_edges, 3);
    }

    #[test]
    fn test_nesting_hierarchy() {
        // 模拟一个大方框和一个小方框在里面
        let outer_points = vec![[0.0, 0.0], [20.0, 0.0], [20.0, 20.0], [0.0, 20.0]];
        let inner_points = vec![[5.0, 5.0], [15.0, 5.0], [15.0, 15.0], [5.0, 15.0]];

        let mut loops = vec![
            Loop {
                points: outer_points,
                is_outer: true,
                signed_area: 400.0,
                nesting_level: 0,
                parent: None,
                children: Vec::new(),
            },
            Loop {
                points: inner_points,
                is_outer: false,
                signed_area: 100.0,
                nesting_level: 0,
                parent: None,
                children: Vec::new(),
            },
        ];

        let stitcher = TopologyStitcher::new();
        stitcher.compute_nesting_hierarchy(&mut loops);

        assert!(loops[1].parent == Some(0));
        assert_eq!(loops[1].nesting_level, 1);
        assert!(loops[0].children.contains(&1));
    }

    #[test]
    fn test_outer_loops() {
        let loops = vec![
            Loop {
                points: vec![[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]],
                is_outer: true,
                signed_area: 100.0,
                nesting_level: 0,
                parent: None,
                children: vec![1],
            },
            Loop {
                points: vec![[2.0, 2.0], [8.0, 2.0], [8.0, 8.0], [2.0, 8.0]],
                is_outer: false,
                signed_area: 36.0,
                nesting_level: 1,
                parent: Some(0),
                children: Vec::new(),
            },
        ];

        let stitcher = TopologyStitcher::new();
        let outer = stitcher.outer_loops(&loops);
        assert_eq!(outer, vec![0]);
    }

    #[test]
    fn test_holes_of_loop() {
        let loops = vec![
            Loop {
                points: vec![[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]],
                is_outer: true,
                signed_area: 100.0,
                nesting_level: 0,
                parent: None,
                children: vec![1],
            },
            Loop {
                points: vec![[2.0, 2.0], [8.0, 2.0], [8.0, 8.0], [2.0, 8.0]],
                is_outer: false,
                signed_area: 36.0,
                nesting_level: 1,
                parent: Some(0),
                children: Vec::new(),
            },
        ];

        let stitcher = TopologyStitcher::new();
        let holes = stitcher.holes_of_loop(&loops, 0);
        assert_eq!(holes, vec![1]);
    }
}
