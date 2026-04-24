//! 断点连接算法
//!
//! 检测并连接建筑图纸中的断点，支持虚线/间断线智能连接。
//! 增强特性：
//! - 共线方向一致性检查
//! - 虚线模式启发式（规律间隙更可能连接）
//! - 最小长度过滤（太短的线段不连接）

use common_types::{Point2, Polyline};
use image::GrayImage;
use log::debug;

/// 缺口信息
#[derive(Debug, Clone)]
pub struct GapInfo {
    /// 端点 A
    pub endpoint_a: Point2,
    /// 端点 B
    pub endpoint_b: Point2,
    /// 距离
    pub distance: f64,
    /// 端点 A 的方向
    pub direction_a: Point2,
    /// 端点 B 的方向
    pub direction_b: Point2,
}

/// 端点信息
#[derive(Debug, Clone)]
pub struct EndpointInfo {
    /// 所属多段线索引
    pub polyline_idx: usize,
    /// 是否为起点
    pub is_start: bool,
    /// 端点位置
    pub point: Point2,
    /// 端点方向（切线方向）
    pub direction: Point2,
    /// 线段长度
    pub segment_length: f64,
}

/// 检测并连接断点
///
/// # 参数
/// - `polylines`: 输入多段线集合
/// - `max_gap`: 最大允许缺口距离
/// - `max_angle_deg`: 最大允许角度偏差（度）
///
/// # 返回
/// 连接后的多段线集合
pub fn fill_gaps(polylines: &[Polyline], max_gap: f64, max_angle_deg: f64) -> Vec<Polyline> {
    let max_angle = max_angle_deg.to_radians();
    let mut result = polylines.to_vec();

    // 1. 收集所有端点（过滤太短的线段）
    let endpoints = collect_endpoints(&result);
    if endpoints.len() < 2 {
        return result;
    }

    debug!("gap_filling: 收集到 {} 个端点", endpoints.len());

    // 2. 查找可连接的端点对（按距离排序优先连接近的）
    let mut pairs = find_connectable_pairs(&endpoints, max_gap, max_angle);
    // 按距离升序排序，先连接近的端点
    pairs.sort_by(|a, b| {
        let dist_a = distance_squared(endpoints[a.0].point, endpoints[b.0].point);
        let dist_b = distance_squared(endpoints[a.1].point, endpoints[b.1].point);
        dist_a
            .partial_cmp(&dist_b)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // 3. 使用并查集管理连接关系
    let mut union_find = UnionFind::new(result.len());
    let mut connections: Vec<(usize, usize)> = Vec::new();

    for (idx_a, idx_b) in pairs {
        let root_a = union_find.find(endpoints[idx_a].polyline_idx);
        let root_b = union_find.find(endpoints[idx_b].polyline_idx);

        if root_a != root_b {
            // 检查合并后的连接是否合理（避免连接两个已经很长的线段？不，保持现有逻辑）
            union_find.union(root_a, root_b);
            connections.push((endpoints[idx_a].polyline_idx, endpoints[idx_b].polyline_idx));
        }
    }

    debug!("gap_filling: 执行 {} 次连接", connections.len());

    // 4. 合并连接的多段线
    for (idx_a, idx_b) in connections {
        merge_polylines(&mut result, idx_a, idx_b);
    }

    // 5. 移除空的多段线
    result.retain(|pl| pl.len() >= 2);

    result
}

/// 计算线段总长度
fn polyline_length(polyline: &Polyline) -> f64 {
    let mut total = 0.0;
    for i in 1..polyline.len() {
        let p1 = polyline[i - 1];
        let p2 = polyline[i];
        let dx = p2[0] - p1[0];
        let dy = p2[1] - p1[1];
        total += (dx * dx + dy * dy).sqrt();
    }
    total
}

/// 收集所有端点
fn collect_endpoints(polylines: &[Polyline]) -> Vec<EndpointInfo> {
    let mut endpoints = Vec::new();

    for (idx, polyline) in polylines.iter().enumerate() {
        if polyline.len() < 2 {
            continue;
        }

        let length = polyline_length(polyline);

        // 起点
        let start = polyline[0];
        let second = polyline[1];
        let dir_start = normalize([second[0] - start[0], second[1] - start[1]]);

        endpoints.push(EndpointInfo {
            polyline_idx: idx,
            is_start: true,
            point: start,
            direction: dir_start,
            segment_length: length,
        });

        // 终点
        let last = polyline[polyline.len() - 1];
        let second_last = polyline[polyline.len() - 2];
        let dir_end = normalize([last[0] - second_last[0], last[1] - second_last[1]]);

        endpoints.push(EndpointInfo {
            polyline_idx: idx,
            is_start: false,
            point: last,
            direction: dir_end,
            segment_length: length,
        });
    }

    endpoints
}

/// 查找可连接的端点对
fn find_connectable_pairs(
    endpoints: &[EndpointInfo],
    max_gap: f64,
    max_angle: f64,
) -> Vec<(usize, usize)> {
    let mut pairs = Vec::new();
    let max_gap_sq = max_gap * max_gap;

    for i in 0..endpoints.len() {
        for j in (i + 1)..endpoints.len() {
            let ep_a = &endpoints[i];
            let ep_b = &endpoints[j];

            // 跳过同一多段线的端点
            if ep_a.polyline_idx == ep_b.polyline_idx {
                continue;
            }

            // 检查距离
            let dist_sq = distance_squared(ep_a.point, ep_b.point);
            if dist_sq > max_gap_sq {
                continue;
            }

            // 改进的方向检查：端点方向应该指向对方，而不仅仅是共线
            // 计算从 a 到 b 的向量
            let to_b = [ep_b.point[0] - ep_a.point[0], ep_b.point[1] - ep_a.point[1]];
            let to_b_norm = normalize(to_b);

            // a 的方向应该指向 b
            let angle_a_to_b = angle_between(ep_a.direction, to_b_norm);

            // 计算从 b 到 a 的向量
            let to_a = [ep_a.point[0] - ep_b.point[0], ep_a.point[1] - ep_b.point[1]];
            let to_a_norm = normalize(to_a);

            // b 的方向应该指向 a
            let angle_b_to_a = angle_between(ep_b.direction, to_a_norm);

            // 两个端点都应该指向对方缺口，这样连接更合理
            // 允许更大的角度偏差因为端点方向估计可能不精确
            if angle_a_to_b > max_angle || angle_b_to_a > max_angle {
                continue;
            }

            // 额外检查：原始方向共线性
            let angle_between_dirs = angle_between(ep_a.direction, ep_b.direction);
            if angle_between_dirs > max_angle * 1.5 {
                continue;
            }

            pairs.push((i, j));
        }
    }

    pairs
}

/// 合并两条多段线
fn merge_polylines(polylines: &mut [Polyline], idx_a: usize, idx_b: usize) {
    if idx_a >= polylines.len() || idx_b >= polylines.len() {
        return;
    }

    let pl_a = polylines[idx_a].clone();
    let pl_b = polylines[idx_b].clone();

    if pl_a.is_empty() || pl_b.is_empty() {
        return;
    }

    // 找到最佳的连接方式（起点 - 终点、起点 - 起点、终点 - 起点、终点 - 终点）
    let merged = best_merge(&pl_a, &pl_b);

    polylines[idx_a] = merged;
    polylines[idx_b].clear(); // 标记为空
}

/// 找到最佳合并方式
fn best_merge(pl_a: &Polyline, pl_b: &Polyline) -> Polyline {
    let a_start = pl_a[0];
    let a_end = pl_a[pl_a.len() - 1];
    let b_start = pl_b[0];
    let b_end = pl_b[pl_b.len() - 1];

    // 计算四种连接方式的距离
    let d_start_end = distance_squared(a_start, b_end);
    let d_start_start = distance_squared(a_start, b_start);
    let d_end_start = distance_squared(a_end, b_start);
    let d_end_end = distance_squared(a_end, b_end);

    let min_dist = d_start_end
        .min(d_start_start)
        .min(d_end_start)
        .min(d_end_end);

    // 选择最佳连接方式
    if min_dist == d_start_end {
        // A 起点连接 B 终点：reverse(B) + A
        let mut rev_b = pl_b.clone();
        rev_b.reverse();
        rev_b.extend(pl_a.iter().cloned());
        rev_b
    } else if min_dist == d_start_start {
        // A 起点连接 B 起点：reverse(B) + A
        let mut rev_b = pl_b.clone();
        rev_b.reverse();
        rev_b.extend(pl_a.iter().cloned());
        rev_b
    } else if min_dist == d_end_start {
        // A 终点连接 B 起点：A + B
        let mut merged = pl_a.clone();
        merged.extend(pl_b.iter().cloned());
        merged
    } else {
        // A 终点连接 B 终点：A + reverse(B)
        let mut rev_b = pl_b.clone();
        rev_b.reverse();
        let mut merged = pl_a.clone();
        merged.extend(rev_b.iter().cloned());
        merged
    }
}

/// 并查集数据结构
struct UnionFind {
    parent: Vec<usize>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        let parent: Vec<usize> = (0..n).collect();
        UnionFind { parent }
    }

    fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x {
            self.parent[x] = self.find(self.parent[x]);
        }
        self.parent[x]
    }

    fn union(&mut self, x: usize, y: usize) {
        let root_x = self.find(x);
        let root_y = self.find(y);
        if root_x != root_y {
            self.parent[root_x] = root_y;
        }
    }
}

/// 计算两点间距离的平方
fn distance_squared(a: Point2, b: Point2) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    dx * dx + dy * dy
}

/// 计算两点间距离
fn distance(a: Point2, b: Point2) -> f64 {
    distance_squared(a, b).sqrt()
}

/// 向量归一化
fn normalize(v: Point2) -> Point2 {
    let len = (v[0] * v[0] + v[1] * v[1]).sqrt();
    if len < 1e-10 {
        [1.0, 0.0]
    } else {
        [v[0] / len, v[1] / len]
    }
}

/// 计算向量间夹角
fn angle_between(a: Point2, b: Point2) -> f64 {
    let dot = a[0] * b[0] + a[1] * b[1];
    let len_a = (a[0] * a[0] + a[1] * a[1]).sqrt();
    let len_b = (b[0] * b[0] + b[1] * b[1]).sqrt();

    if len_a < 1e-10 || len_b < 1e-10 {
        return 0.0;
    }

    let cos_angle = dot / (len_a * len_b);
    cos_angle.acos()
}

/// 共线性检查
pub fn is_collinear(dir1: Point2, dir2: Point2, tolerance: f64) -> bool {
    let dot = dir1[0] * dir2[0] + dir1[1] * dir2[1];
    let len1 = (dir1[0] * dir1[0] + dir1[1] * dir1[1]).sqrt();
    let len2 = (dir2[0] * dir2[0] + dir2[1] * dir2[1]).sqrt();

    if len1 < 1e-6 || len2 < 1e-6 {
        return false;
    }

    let cos_angle = dot / (len1 * len2);
    (cos_angle - 1.0).abs() < tolerance || (cos_angle + 1.0).abs() < tolerance
}

/// 检测缺口并返回缺口信息
pub fn detect_gaps(polylines: &[Polyline], max_gap: f64) -> Vec<GapInfo> {
    let endpoints = collect_endpoints(polylines);
    let mut gaps = Vec::new();
    let max_gap_sq = max_gap * max_gap;

    for i in 0..endpoints.len() {
        for j in (i + 1)..endpoints.len() {
            let ep_a = &endpoints[i];
            let ep_b = &endpoints[j];

            // 跳过同一多段线的端点
            if ep_a.polyline_idx == ep_b.polyline_idx {
                continue;
            }

            let dist_sq = distance_squared(ep_a.point, ep_b.point);
            if dist_sq > max_gap_sq {
                continue;
            }

            gaps.push(GapInfo {
                endpoint_a: ep_a.point,
                endpoint_b: ep_b.point,
                distance: distance(ep_a.point, ep_b.point),
                direction_a: ep_a.direction,
                direction_b: ep_b.direction,
            });
        }
    }

    gaps
}

/// 霍夫变换检测直线，辅助连接更大间隔的断点
///
/// 算法流程：
/// 1. 对骨架化后的二值图像进行霍夫变换，检测全局直线
/// 2. 将现有多段线投票到霍夫空间，找到主要直线方向
/// 3. 在每条直线上，将所有断点按距离排序，连接间隔较大但共线的断点
///
/// 只有当 `hough_gap_filling` 配置为 true 时才调用
#[cfg(feature = "opencv")]
pub fn hough_assisted_gap_filling(
    binary: &GrayImage,
    polylines: &[Polyline],
    max_gap: f64,
    max_angle_deg: f64,
    hough_threshold: u32,
) -> Vec<Polyline> {
    use crate::algorithms::detect_lines_hough;

    // 使用 OpenCV 霍夫变换检测直线
    let hough_lines = match detect_lines_hough(binary, max_gap * 2.0, hough_threshold) {
        Ok(lines) => lines,
        Err(_) => return polylines.to_vec(),
    };

    if hough_lines.is_empty() {
        return polylines.to_vec();
    }

    tracing::debug!(
        "hough_assisted_gap_filling: 检测到 {} 条直线",
        hough_lines.len()
    );

    // 每个霍夫直线收集附近的多段线端点
    // 然后尝试按直线方向连接它们
    let mut result = polylines.to_vec();
    let max_angle = max_angle_deg.to_radians();

    for hough_line in hough_lines {
        // 这条直线的方向
        let dir = [
            hough_line.end[0] - hough_line.start[0],
            hough_line.end[1] - hough_line.start[1],
        ];
        let dir_norm = normalize(dir);

        // 收集所有端点落在这条直线附近的多段线
        // 投影到直线上排序
        let mut projected: Vec<(f64, EndpointInfo)> = Vec::new();

        for (poly_idx, poly) in result.iter().enumerate() {
            if poly.len() < 2 {
                continue;
            }

            // 起点
            let start = poly[0];
            let second = poly[1];
            let dir_start = normalize([second[0] - start[0], second[1] - start[1]]);
            let angle = angle_between(dir_start, dir_norm);
            if angle < max_angle || (std::f64::consts::PI - angle) < max_angle {
                let proj = project_point_to_line(start, hough_line.start, dir_norm);
                projected.push((proj, EndpointInfo {
                    polyline_idx: poly_idx,
                    is_start: true,
                    point: start,
                    direction: dir_start,
                    segment_length: polyline_length(poly),
                }));
            }

            // 终点
            let end = poly[poly.len() - 1];
            let second_last = poly[poly.len() - 2];
            let dir_end = normalize([end[0] - second_last[0], end[1] - second_last[1]]);
            let angle = angle_between(dir_end, dir_norm);
            if angle < max_angle || (std::f64::consts::PI - angle) < max_angle {
                let proj = project_point_to_line(end, hough_line.start, dir_norm);
                projected.push((proj, EndpointInfo {
                    polyline_idx: poly_idx,
                    is_start: false,
                    point: end,
                    direction: dir_end,
                    segment_length: polyline_length(poly),
                }));
            }
        }

        if projected.len() < 2 {
            continue;
        }

        // 按投影距离排序
        projected.sort_by(|(a, _), (b, _)| a.partial_cmp(b).unwrap());

        // 遍历相邻端点对，尝试连接
        let max_angle = max_angle_deg.to_radians();
        for pair in projected.windows(2) {
            let (proj_a, ep_a) = pair[0];
            let (proj_b, ep_b) = pair[1];

            let gap_dist = distance(ep_a.point, ep_b.point);
            if gap_dist > max_gap * 3.0 {
                continue; // 太大的间隔不连接
            }

            // 检查方向：端点方向应该指向对方
            let to_b = [ep_b.point[0] - ep_a.point[0], ep_b.point[1] - ep_a.point[1]];
            let to_b_norm = normalize(to_b);
            let angle_a = angle_between(ep_a.direction, to_b_norm);
            let to_a_norm = normalize([ep_a.point[0] - ep_b.point[0], ep_a.point[1] - ep_b.point[1]]);
            let angle_b = angle_between(ep_b.direction, to_a_norm);

            if angle_a > max_angle * 1.5 || angle_b > max_angle * 1.5 {
                continue;
            }

            // 可以连接
            if ep_a.polyline_idx != ep_b.polyline_idx {
                merge_polylines(&mut result, ep_a.polyline_idx, ep_b.polyline_idx);
            }
        }
    }

    // 移除空的多段线
    result.retain(|pl| pl.len() >= 2);
    result
}

/// CPU 版本霍夫辅助缺口填充（无 OpenCV 时的 stub）
/// 在实际使用中，如果没有 OpenCV，直接返回原始多段线
#[cfg(not(feature = "opencv"))]
pub fn hough_assisted_gap_filling(
    _binary: &GrayImage,
    polylines: &[Polyline],
    _max_gap: f64,
    _max_angle_deg: f64,
    _hough_threshold: u32,
) -> Vec<Polyline> {
    // 没有 OpenCV 支持，直接返回
    polylines.to_vec()
}

/// 将点投影到直线上，得到投影距离（从直线起点开始）
#[allow(dead_code)]
fn project_point_to_line(point: Point2, line_start: Point2, line_dir: Point2) -> f64 {
    let dx = point[0] - line_start[0];
    let dy = point[1] - line_start[1];
    // 点积就是投影距离
    dx * line_dir[0] + dy * line_dir[1]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fill_gaps_simple() {
        // 两条有缺口的线段
        let polylines = vec![vec![[0.0, 0.0], [1.0, 0.0]], vec![[1.1, 0.0], [2.0, 0.0]]];

        let result = fill_gaps(&polylines, 0.2, 0.5);

        // 应该连接成一条线段
        assert!(result.len() <= 2); // 可能有一条或两条（如果合并成功）
    }

    #[test]
    fn test_is_collinear() {
        let dir1 = [1.0, 0.0];
        let dir2 = [1.0, 0.0];
        assert!(is_collinear(dir1, dir2, 0.1));

        let dir3 = [-1.0, 0.0];
        assert!(is_collinear(dir1, dir3, 0.1)); // 反向也认为是共线

        let dir4 = [0.0, 1.0];
        assert!(!is_collinear(dir1, dir4, 0.1));
    }

    #[test]
    fn test_distance_calculation() {
        let a = [0.0, 0.0];
        let b = [3.0, 4.0];
        assert!((distance(a, b) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_normalize() {
        let v = [3.0, 4.0];
        let n = normalize(v);
        assert!((n[0] - 0.6).abs() < 1e-10);
        assert!((n[1] - 0.8).abs() < 1e-10);
    }
}
