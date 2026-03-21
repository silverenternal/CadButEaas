//! 闭合环提取器 - 支持非连通图和悬挂边处理
//!
//! 改进：
//! 1. 能够处理非连通图（多个独立的环）
//! 2. 能够处理悬挂边（不构成环的边会被跳过）
//! 3. 使用 Hierholzer 算法的变体来提取所有环

use common_types::{Point2, ClosedLoop};
use std::collections::{HashMap, HashSet};

/// 环提取器 - 从平面图中提取闭合环
pub struct LoopExtractor {
    /// 容差（用于后续过滤退化环）
    #[allow(dead_code)] // 保留用于未来扩展
    tolerance: f64,
    /// 最小环面积（小于此值的环会被过滤）
    min_area: f64,
}

impl LoopExtractor {
    pub fn new(tolerance: f64) -> Self {
        Self {
            tolerance,
            min_area: tolerance * tolerance,
        }
    }

    pub fn with_min_area(mut self, min_area: f64) -> Self {
        self.min_area = min_area;
        self
    }

    /// 从点和边中提取所有闭合环
    ///
    /// 改进：
    /// - 处理非连通图：遍历所有未使用的边，每个连通分量都能被处理
    /// - 处理悬挂边：无法形成闭合路径的边会被跳过而不是失败
    pub fn extract_loops(&self, points: &[Point2], edges: &[(usize, usize)]) -> Vec<ClosedLoop> {
        if points.is_empty() || edges.is_empty() {
            return Vec::new();
        }

        // 构建邻接表（使用多重边支持）
        let mut adjacency: HashMap<usize, Vec<usize>> = HashMap::new();
        for &(a, b) in edges {
            adjacency.entry(a).or_default().push(b);
            adjacency.entry(b).or_default().push(a);
        }

        // 跟踪每条边的使用次数（支持多重边）
        let mut edge_usage: HashMap<(usize, usize), usize> = HashMap::new();
        for &(a, b) in edges {
            *edge_usage.entry((a, b)).or_insert(0) += 1;
            *edge_usage.entry((b, a)).or_insert(0) += 1;
        }

        let mut loops = Vec::new();
        let mut used_edges: HashSet<(usize, usize)> = HashSet::new();

        // 遍历所有边，处理非连通图
        for &start_edge in edges {
            // 检查边是否已被使用（双向检查）
            if used_edges.contains(&start_edge) || used_edges.contains(&(start_edge.1, start_edge.0)) {
                continue;
            }

            // 尝试从这条边开始追踪环
            if let Some(loop_indices) = self.trace_loop_eulerian(
                points,
                &adjacency,
                start_edge,
                &mut used_edges,
                &edge_usage,
            ) {
                if loop_indices.len() >= 3 {
                    // 将索引转换为点
                    let loop_points: Vec<Point2> = loop_indices
                        .iter()
                        .filter_map(|&idx| points.get(idx).copied())
                        .collect();
                    
                    if loop_points.len() >= 3 {
                        let signed_area = calculate_signed_area(&loop_points);
                        if signed_area.abs() > self.min_area {
                            loops.push(ClosedLoop {
                                points: loop_points,
                                signed_area,
                            });
                        }
                    }
                }
            }
        }

        // 按面积排序（最大的通常是外轮廓）
        loops.sort_by(|a, b| b.signed_area.abs().partial_cmp(&a.signed_area.abs()).unwrap());

        loops
    }

    /// 追踪单个环（改进版：处理非欧拉图）
    ///
    /// 使用启发式方法：
    /// 1. 优先选择"最直"的路径（角度连续性）
    /// 2. 如果无法回到起点，回溯并尝试其他路径
    /// 3. 支持处理 T 型连接和悬挂边
    fn trace_loop_eulerian(
        &self,
        points: &[Point2],
        adjacency: &HashMap<usize, Vec<usize>>,
        start_edge: (usize, usize),
        used_edges: &mut HashSet<(usize, usize)>,
        edge_usage: &HashMap<(usize, usize), usize>,
    ) -> Option<Vec<usize>> {
        // 检查起始边是否可用
        if !is_edge_available(start_edge, used_edges, edge_usage) {
            return None;
        }

        // 使用回溯法查找环
        let mut path = vec![start_edge.0];
        let mut current = start_edge.1;
        let mut prev = start_edge.0;

        // 标记起始边为已使用
        used_edges.insert((prev, current));

        // 深度优先搜索找环
        loop {
            path.push(current);

            // 检查是否回到起点
            if current == start_edge.0 && path.len() > 2 {
                // 成功找到环
                path.pop(); // 移除重复的终点
                return Some(path);
            }

            // 找到下一个点
            let neighbors = adjacency.get(&current)?;
            
            // 选择下一个点（基于角度和可用性）
            let next = self.select_next_point_with_backtrack(
                points,
                current,
                prev,
                neighbors,
                used_edges,
                edge_usage,
                &mut path,
            );

            match next {
                Some(n) => {
                    used_edges.insert((current, n));
                    prev = current;
                    current = n;
                }
                None => {
                    // 无法继续，回溯
                    if path.len() <= 1 {
                        return None; // 完全无法形成环
                    }
                    
                    // 回溯到上一个点
                    used_edges.remove(&(prev, current));
                    path.pop();
                    current = prev;
                    prev = path.last().copied()?;
                }
            }

            // 防止无限循环
            if path.len() > points.len() + 1 {
                return None;
            }
        }
    }

    /// 选择下一个点（带回溯支持）
    #[allow(clippy::too_many_arguments, clippy::ptr_arg)]
    fn select_next_point_with_backtrack(
        &self,
        points: &[Point2],
        current: usize,
        prev: usize,
        candidates: &[usize],
        used_edges: &HashSet<(usize, usize)>,
        edge_usage: &HashMap<(usize, usize), usize>,
        path: &mut Vec<usize>,
    ) -> Option<usize> {
        let current_point = points[current];
        let prev_point = points[prev];

        // 入射向量
        let incoming = [
            current_point[0] - prev_point[0],
            current_point[1] - prev_point[1],
        ];

        // 收集所有可用的下一个点，按角度分数排序
        let mut available: Vec<(usize, f64)> = Vec::new();

        for &next in candidates {
            // 检查边是否可用
            if !is_edge_available((current, next), used_edges, edge_usage) {
                continue;
            }

            // 避免立即回到前一个点（除非是起点且路径足够长）
            if next == prev && path.len() > 2 {
                continue;
            }

            // 避免访问路径中已有的点（除非是起点）
            if next != start_node(path) && path.contains(&next) {
                continue;
            }

            let next_point = points[next];
            let outgoing = [
                next_point[0] - current_point[0],
                next_point[1] - current_point[1],
            ];

            let score = self.angle_score(&incoming, &outgoing);
            available.push((next, score));
        }

        // 按分数降序排序
        available.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // 返回最佳选择
        available.first().map(|(next, _)| *next)
    }

    /// 计算角度分数
    fn angle_score(&self, v1: &[f64; 2], v2: &[f64; 2]) -> f64 {
        let len1 = (v1[0] * v1[0] + v1[1] * v1[1]).sqrt();
        let len2 = (v2[0] * v2[0] + v2[1] * v2[1]).sqrt();

        if len1 < 1e-10 || len2 < 1e-10 {
            return 0.0;
        }

        // 归一化
        let v1_norm = [v1[0] / len1, v1[1] / len1];
        let v2_norm = [v2[0] / len2, v2[1] / len2];

        // 点积 = cos(θ)，越大表示角度越小（越直）
        v1_norm[0] * v2_norm[0] + v1_norm[1] * v2_norm[1]
    }
}

/// 检查边是否可用
fn is_edge_available(
    edge: (usize, usize),
    used_edges: &HashSet<(usize, usize)>,
    edge_usage: &HashMap<(usize, usize), usize>,
) -> bool {
    let (a, b) = edge;
    let reverse = (b, a);

    // 获取边的总使用次数
    let total_usage = edge_usage.get(&edge).copied().unwrap_or(1);

    // 计算已使用次数（双向）- used_edges 是 HashSet，所以检查是否存在
    let used_count = if used_edges.contains(&edge) { 1 } else { 0 }
        + if used_edges.contains(&reverse) { 1 } else { 0 };

    used_count < total_usage
}

/// 获取路径的起始节点
fn start_node(path: &[usize]) -> usize {
    path.first().copied().unwrap_or(0)
}

/// 计算多边形的有符号面积
fn calculate_signed_area(points: &[Point2]) -> f64 {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_rectangle() {
        let points = vec![
            [0.0, 0.0],
            [10.0, 0.0],
            [10.0, 10.0],
            [0.0, 10.0],
        ];
        let edges = vec![(0, 1), (1, 2), (2, 3), (3, 0)];

        let extractor = LoopExtractor::new(0.5);
        let loops = extractor.extract_loops(&points, &edges);

        assert_eq!(loops.len(), 1);
        assert!((loops[0].signed_area - 100.0).abs() < 1e-10);
    }

    #[test]
    fn test_extract_multiple_disconnected_loops() {
        // 两个不相连的矩形
        let points = vec![
            // 矩形 1
            [0.0, 0.0],
            [10.0, 0.0],
            [10.0, 10.0],
            [0.0, 10.0],
            // 矩形 2
            [20.0, 0.0],
            [30.0, 0.0],
            [30.0, 10.0],
            [20.0, 10.0],
        ];
        let edges = vec![
            (0, 1), (1, 2), (2, 3), (3, 0), // 矩形 1
            (4, 5), (5, 6), (6, 7), (7, 4), // 矩形 2
        ];

        let extractor = LoopExtractor::new(0.5);
        let loops = extractor.extract_loops(&points, &edges);

        assert_eq!(loops.len(), 2);
    }

    #[test]
    fn test_extract_with_dangling_edge() {
        // 一个矩形加一条悬挂边
        let points = vec![
            [0.0, 0.0],
            [10.0, 0.0],
            [10.0, 10.0],
            [0.0, 10.0],
            [15.0, 15.0], // 悬挂点
        ];
        let edges = vec![
            (0, 1), (1, 2), (2, 3), (3, 0), // 矩形
            (1, 4), // 悬挂边
        ];

        let extractor = LoopExtractor::new(0.5);
        let loops = extractor.extract_loops(&points, &edges);

        // 应该只提取矩形环，忽略悬挂边
        assert_eq!(loops.len(), 1);
    }

    #[test]
    fn test_signed_area_orientation() {
        // 逆时针（正面积）
        let ccw = vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        assert!(calculate_signed_area(&ccw) > 0.0);

        // 顺时针（负面积）
        let cw = vec![[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]];
        assert!(calculate_signed_area(&cw) < 0.0);
    }

    #[test]
    fn test_extract_with_hole() {
        // 外矩形和内矩形（孔洞）
        let points = vec![
            // 外矩形
            [0.0, 0.0],
            [20.0, 0.0],
            [20.0, 20.0],
            [0.0, 20.0],
            // 内矩形（孔洞）
            [5.0, 5.0],
            [15.0, 5.0],
            [15.0, 15.0],
            [5.0, 15.0],
        ];
        let edges = vec![
            (0, 1), (1, 2), (2, 3), (3, 0), // 外矩形
            (4, 5), (5, 6), (6, 7), (7, 4), // 内矩形
        ];

        let extractor = LoopExtractor::new(0.5);
        let loops = extractor.extract_loops(&points, &edges);

        assert_eq!(loops.len(), 2);
        // 外矩形面积应该是 400，内矩形是 100
        assert!(loops[0].signed_area.abs() > loops[1].signed_area.abs());
    }
}
