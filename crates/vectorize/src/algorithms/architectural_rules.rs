//! 建筑图纸规则几何校正
//!
//! 基于建筑图纸的常见规则进行几何后处理：
//! 1. 正交性校正：检测主要方向（X/Y），将近乎正交的线段调整到精确正交
//! 2. 平行线段分组：检测平行且等距的线段（通常是墙线/梁），调整间距使其均匀
//! 3. 尝试闭合开放墙线，形成完整房间轮廓

use common_types::{Point2, Polyline};
use log::debug;

/// 正交性校正 - 将线段对齐到主要正交方向
///
/// 算法：
/// 1. 统计所有线段的角度，找出两个主峰（通常是 0° 和 90°）
/// 2. 将接近主峰角度的线段旋转对齐到精确角度
///
/// 容差：角度偏差小于 `tolerance_deg` 才会被校正
pub fn orthogonality_correction(polylines: &mut [Polyline], tolerance_deg: f64) {
    let tolerance_rad = tolerance_deg.to_radians();

    // 统计所有线段的角度直方图
    // 分辨率：180 bins 每个 bin 对应 1 度
    let mut histogram = [0usize; 180];

    // 收集所有线段角度
    for poly in polylines.iter() {
        if poly.len() < 2 {
            continue;
        }

        // 对每条线段计算整体方向
        let (start, end) = (poly[0], poly[poly.len() - 1]);
        let dx = end[0] - start[0];
        let dy = end[1] - start[1];
        let angle = dy.atan2(dx); // [-pi, pi]
        let mut angle_deg = angle.to_degrees();
        if angle_deg < 0.0 {
            angle_deg += 180.0;
        }

        let bin = angle_deg.floor() as usize;
        histogram[bin % 180] += 1;
    }

    // 找出两个主峰（局部最大值）
    let mut peaks = find_peaks(&histogram, 3);
    if peaks.is_empty() {
        return;
    }

    // 如果只有一个峰，再找一个接近 90 度的
    if peaks.len() == 1 {
        let center = peaks[0].0;
        // 找 90 度附近的峰
        let target = (center + 90) % 180;
        let mut best_count = 0;
        let mut best_idx = target;
        for offset in -5..=5 {
            let idx = (target as i32 + offset).rem_euclid(180) as usize;
            if histogram[idx] > best_count {
                best_count = histogram[idx];
                best_idx = idx;
            }
        }
        if best_count > 0 {
            peaks.push((best_idx, best_count));
        }
    }

    debug!(
        "orthogonality_correction: 检测到 {} 个主峰: {:?}",
        peaks.len(),
        peaks
    );

    // 将峰转换为弧度
    let mut main_directions: Vec<f64> = peaks
        .iter()
        .map(|&(bin, _)| (bin as f64 + 0.5).to_radians())
        .collect();

    // 如果方向差接近 90 度，调整到精确正交
    if main_directions.len() >= 2 {
        let diff = (main_directions[0] - main_directions[1]).abs();
        if (diff - std::f64::consts::FRAC_PI_2).abs() < 0.3 {
            // 调整到精确 0 度和 90 度
            let (a0, a1) = if main_directions[0].abs() < std::f64::consts::FRAC_PI_4 {
                (0.0, std::f64::consts::FRAC_PI_2)
            } else {
                (std::f64::consts::FRAC_PI_2, 0.0)
            };
            main_directions[0] = a0;
            main_directions[1] = a1;
            debug!("orthogonality_correction: 调整为精确正交 0°/90°");
        }
    }

    // 对每个多段线中的每个线段，找到最近的主方向，旋转对齐
    for poly in polylines {
        if poly.len() < 2 {
            continue;
        }

        // 整体方向决定这个多段线最接近哪个主方向
        let (start, end) = (poly[0], poly[poly.len() - 1]);
        let dx = end[0] - start[0];
        let dy = end[1] - start[1];
        let angle = dy.atan2(dx);

        // 找到最近的主方向
        let (closest_dir, _min_diff) = main_directions
            .iter()
            .map(|&dir| {
                let diff = (angle - dir).abs();
                let diff = diff.min(std::f64::consts::PI - diff);
                (dir, diff)
            })
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap_or((angle, 0.0));

        // 如果偏差在容差内，旋转整个多段线
        // 整个多段线绕起点旋转
        let diff = (closest_dir - angle).abs();
        if diff < tolerance_rad || (diff - std::f64::consts::PI).abs() < tolerance_rad {
            let rotation = closest_dir - angle;
            rotate_polyline(poly, rotation, start);
        }
    }
}

/// 旋转多段线绕指定起点
fn rotate_polyline(poly: &mut Polyline, rotation: f64, center: Point2) {
    let cos = rotation.cos();
    let sin = rotation.sin();

    for point in poly {
        let dx = point[0] - center[0];
        let dy = point[1] - center[1];

        let new_dx = dx * cos - dy * sin;
        let new_dy = dx * sin + dy * cos;

        point[0] = center[0] + new_dx;
        point[1] = center[1] + new_dy;
    }
}

/// 在直方图中找局部峰值
/// 返回 (bin_index, count)
fn find_peaks(histogram: &[usize; 180], window: usize) -> Vec<(usize, usize)> {
    let mut peaks = Vec::new();

    for i in 0..180 {
        let count = histogram[i];
        let mut is_peak = true;

        // 检查窗口内是否有更大的值
        for offset in -(window as i32)..=window as i32 {
            if offset == 0 {
                continue;
            }
            let j = (i as i32 + offset).rem_euclid(180) as usize;
            if histogram[j] > count {
                is_peak = false;
                break;
            }
        }

        if is_peak && count > 0 {
            peaks.push((i, count));
        }
    }

    // 按票数降序排序
    peaks.sort_by_key(|p| std::cmp::Reverse(p.1));
    peaks
}

/// 平行线段分组校正
///
/// 检测近似平行且近似等距的线段（通常是平行墙线/梁柱），
/// 调整它们的间距使其均匀分布，更符合建筑设计规则
pub fn parallel_uniform_spacing(
    polylines: &mut [Polyline],
    angle_tolerance_deg: f64,
    spacing_tolerance: f64,
) {
    if polylines.len() < 3 {
        return;
    }

    // 按角度分组 - 相近角度分为一组
    let angle_tolerance = angle_tolerance_deg.to_radians();
    let mut groups: Vec<Vec<usize>> = Vec::new();
    let mut group_angles: Vec<f64> = Vec::new();

    for (idx, poly) in polylines.iter().enumerate() {
        if poly.len() < 2 {
            continue;
        }

        // 计算线段整体方向角度 [0, pi)
        let (start, end) = (poly[0], poly[poly.len() - 1]);
        let dx = end[0] - start[0];
        let dy = end[1] - start[1];
        let mut angle = dy.atan2(dx);
        if angle < 0.0 {
            angle += std::f64::consts::PI; // 归一化到 [0, pi)
        }

        // 找一个现有组
        let mut found = false;
        for (group_idx, &group_angle) in group_angles.iter().enumerate() {
            let diff = (angle - group_angle).abs();
            let diff = diff.min(std::f64::consts::PI - diff);
            if diff < angle_tolerance {
                groups[group_idx].push(idx);
                found = true;
                break;
            }
        }

        if !found {
            groups.push(vec![idx]);
            group_angles.push(angle);
        }
    }

    // 对每个足够大的组，进行均匀间距调整
    for group in groups {
        if group.len() >= 3 {
            correct_group_parallel_spacing(&group[..], polylines, spacing_tolerance);
        }
    }
}

/// 对一组平行线进行均匀间距校正
fn correct_group_parallel_spacing(group: &[usize], polylines: &mut [Polyline], tolerance: f64) {
    // 计算每条线的平均位置（沿垂直方向）
    // 对于平行线，我们将它们投影到垂直方向，排序
    let mut positions: Vec<(f64, usize)> = group
        .iter()
        .map(|&idx| {
            let poly = &polylines[idx];
            let center = poly_center(poly);
            // 计算投影到垂直方向：(dx, dy) 是线段方向，垂直方向是 (-dy, dx)
            let (start, end) = (poly[0], poly[poly.len() - 1]);
            let dx = end[0] - start[0];
            let dy = end[1] - start[1];
            let proj = center[0] * (-dy) + center[1] * dx; // 点积就是投影
            (proj, idx)
        })
        .collect();

    // 按投影位置排序
    positions.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    // 计算平均间距
    let first = positions.first().unwrap().0;
    let last = positions.last().unwrap().0;
    let expected_spacing = (last - first) / (group.len() - 1) as f64;

    // 检查间距是否已经接近均匀
    let mut is_uniform = true;
    for window in positions.windows(2) {
        let spacing = window[1].0 - window[0].0;
        if (spacing - expected_spacing).abs() > expected_spacing * tolerance {
            is_uniform = false;
            break;
        }
    }

    if is_uniform {
        return; // 已经均匀，不需要调整
    }

    debug!(
        "parallel_uniform_spacing: 校正 {} 条平行线，平均间距 {:.2}",
        group.len(),
        expected_spacing
    );

    // 调整每条线位置，使间距均匀
    for (i, &(_, poly_idx)) in positions.iter().enumerate() {
        let expected_pos = first + i as f64 * expected_spacing;
        let current_pos = positions[i].0;
        let offset = expected_pos - current_pos;

        // 移动整个线段 offset 垂直方向
        let poly = &mut polylines[poly_idx];
        let (start, end) = (poly[0], poly[poly.len() - 1]);
        let dx = end[0] - start[0];
        let dy = end[1] - start[1];
        let len = (dx * dx + dy * dy).sqrt();
        if len < 1e-6 {
            continue;
        }
        // 单位垂直向量
        let vx = -dy / len;
        let vy = dx / len;

        // 投影方向向量 (-dy, dx) 长度就是 len，所以 offset 需要除以 len 得到实际坐标偏移
        let actual_offset = offset / len;

        // 移动所有点
        for point in poly {
            point[0] += vx * actual_offset;
            point[1] += vy * actual_offset;
        }
    }
}

/// 计算多段线中心点
fn poly_center(poly: &Polyline) -> Point2 {
    let cx = poly.iter().map(|p| p[0]).sum::<f64>() / poly.len() as f64;
    let cy = poly.iter().map(|p| p[1]).sum::<f64>() / poly.len() as f64;
    [cx, cy]
}

/// 计算两点间距离的平方
fn distance_squared(a: Point2, b: Point2) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    dx * dx + dy * dy
}

/// 端点信息
#[derive(Debug, Clone)]
struct EndpointInfo {
    /// 所属多段线索引
    pub polyline_idx: usize,
    /// 端点位置
    pub point: Point2,
    /// 端点方向（切线方向）
    pub direction: Point2,
}

/// 并查集数据结构
struct UnionFind {
    parent: Vec<usize>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
        }
    }

    fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x {
            self.parent[x] = self.find(self.parent[x]);
        }
        self.parent[x]
    }

    fn union(&mut self, x: usize, y: usize) {
        let rx = self.find(x);
        let ry = self.find(y);
        if rx != ry {
            self.parent[rx] = ry;
        }
    }
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

/// 向量归一化
fn normalize(v: Point2) -> Point2 {
    let len = (v[0] * v[0] + v[1] * v[1]).sqrt();
    if len < 1e-10 {
        [1.0, 0.0]
    } else {
        [v[0] / len, v[1] / len]
    }
}

/// 收集所有端点
fn collect_endpoints(polylines: &[Polyline]) -> Vec<EndpointInfo> {
    let mut endpoints = Vec::new();

    for (idx, polyline) in polylines.iter().enumerate() {
        if polyline.len() < 2 {
            continue;
        }

        let _length = polyline_length(polyline);

        // 起点
        let start = polyline[0];
        let second = polyline[1];
        let dir_start = normalize([second[0] - start[0], second[1] - start[1]]);

        endpoints.push(EndpointInfo {
            polyline_idx: idx,
            point: start,
            direction: dir_start,
        });

        // 终点
        let last = polyline[polyline.len() - 1];
        let second_last = polyline[polyline.len() - 2];
        let dir_end = normalize([last[0] - second_last[0], last[1] - second_last[1]]);

        endpoints.push(EndpointInfo {
            polyline_idx: idx,
            point: last,
            direction: dir_end,
        });
    }

    endpoints
}

/// 尝试闭合开放的轮廓
/// 找出接近的端点可以连接形成闭合房间轮廓
/// 对于建筑图纸，墙线应该闭合形成房间
pub fn close_open_contours(polylines: &mut Vec<Polyline>, max_gap: f64, max_angle_deg: f64) {
    let max_angle = max_angle_deg.to_radians();
    let endpoints = collect_endpoints(polylines);

    if endpoints.len() < 2 {
        return;
    }

    // 找起点和终点接近的配对，属于不同多段线，方向接近相对，距离近
    let mut union_find = UnionFind::new(polylines.len());
    let mut connections = Vec::new();

    for i in 0..endpoints.len() {
        for j in (i + 1)..endpoints.len() {
            let ep_a = &endpoints[i];
            let ep_b = &endpoints[j];

            if ep_a.polyline_idx == ep_b.polyline_idx {
                continue; // 同一多段线跳过
            }

            let dist_sq = distance_squared(ep_a.point, ep_b.point);
            if dist_sq > max_gap * max_gap {
                continue;
            }

            // 闭合轮廓：两个端点方向应该接近相反（指向对方）
            let angle_cos = (ep_a.direction[0] * ep_b.direction[0]
                + ep_a.direction[1] * ep_b.direction[1])
                .abs();
            // cos 接近 1 说明方向相反
            if (angle_cos - 1.0).abs() < max_angle.sin() {
                union_find.union(ep_a.polyline_idx, ep_b.polyline_idx);
                connections.push((ep_a.polyline_idx, ep_b.polyline_idx));
            }
        }
    }

    if connections.is_empty() {
        return;
    }

    debug!(
        "close_open_contours: 尝试连接 {} 对端点闭合轮廓",
        connections.len()
    );

    // 合并连接的多段线，闭合轮廓
    // 需要特殊处理可变借用，我们从后往前处理保证索引不失效
    let mut sorted_connections = connections;
    sorted_connections.sort_by_key(|c| std::cmp::Reverse(c.0));

    for (idx_a, idx_b) in sorted_connections {
        if polylines[idx_a].len() >= 2 && polylines[idx_b].len() >= 2 {
            let (a, b) = if idx_a < idx_b {
                let (left, right) = polylines.split_at_mut(idx_b);
                (&mut left[idx_a], &mut right[0])
            } else {
                let (left, right) = polylines.split_at_mut(idx_a);
                (&mut right[0], &mut left[idx_b])
            };
            merge_and_close_polylines(a, b);
        }
    }

    // 移除空的
    polylines.retain(|pl| pl.len() >= 2);
}

/// 合并两个多段线并尝试闭合
/// 假设两个端点已经接近要连接，连接后闭合轮廓
fn merge_and_close_polylines(a: &mut Polyline, b: &mut Polyline) {
    // 四种连接方式：
    // a 的一个端点连接 b 的一个端点，然后闭合
    // 我们已经知道哪两个端点要连接，直接连接并闭合
    let a_start = a[0];
    let a_end = a[a.len() - 1];
    let b_start = b[0];
    let b_end = b[b.len() - 1];

    // 找到哪两个端点最近
    let dists = [
        (distance_squared(a_start, b_start), 0),
        (distance_squared(a_start, b_end), 1),
        (distance_squared(a_end, b_start), 2),
        (distance_squared(a_end, b_end), 3),
    ];

    let (_, &(_min_dist, min_idx)) = dists
        .iter()
        .enumerate()
        .min_by(|(_, (d1, _)), (_, (d2, _))| d1.partial_cmp(d2).unwrap())
        .unwrap();

    let mut merged = match min_idx {
        0 => {
            // a_start connected to b_start: reverse b then + a
            let mut rev_b = b.clone();
            rev_b.reverse();
            rev_b.extend(a.iter().cloned());
            rev_b
        }
        1 => {
            // a_start connected to b_end: reverse b then + a
            let mut rev_b = b.clone();
            rev_b.reverse();
            rev_b.extend(a.iter().cloned());
            rev_b
        }
        2 => {
            // a_end connected to b_start: a + b
            let mut res = a.clone();
            res.extend(b.iter().cloned());
            res
        }
        _ => {
            // a_end connected to b_end: a + reverse b
            let mut res = a.clone();
            let mut rev_b = b.clone();
            rev_b.reverse();
            res.extend(rev_b.iter().cloned());
            res
        }
    };

    // 如果起点终点已经连接，闭合轮廓：添加起点到终点
    if let Some(first) = merged.first() {
        if merged.last() != Some(first) {
            merged.push(*first);
        }
    }

    *a = merged;
    b.clear();
}

/// 完整的建筑规则校正流水线
pub fn correct_all(
    polylines: &mut Vec<Polyline>,
    ortho_tolerance_deg: f64,
    parallel_tolerance: f64,
    close_max_gap: f64,
) {
    if ortho_tolerance_deg > 0.0 {
        orthogonality_correction(polylines, ortho_tolerance_deg);
    }
    if parallel_tolerance > 0.0 {
        parallel_uniform_spacing(polylines, 5.0, parallel_tolerance);
    }
    if close_max_gap > 0.0 {
        close_open_contours(polylines, close_max_gap, 10.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_orthogonality_correction() {
        // 两条近似正交线段，稍微有点偏差
        let mut polylines = vec![
            vec![[0.0, 1.0], [100.0, 2.0]],   // 近似水平 (0.57 度，接近 0)
            vec![[50.0, 0.0], [51.0, 100.0]], // 近似垂直，89.4 度接近 90
        ];

        orthogonality_correction(&mut polylines, 3.0);

        // 第一条应该变成精确水平，y 坐标相同
        let dy = polylines[0][0][1] - polylines[0][1][1];
        assert!(dy.abs() < 0.1);

        // 第二条应该变成精确垂直，x 坐标相同
        let dx = polylines[1][0][0] - polylines[1][1][0];
        assert!(dx.abs() < 0.1);
    }

    #[test]
    fn test_uniform_spacing() {
        // 三条近似平行水平线，间距不均匀
        // 间距 10 和 18 → 平均 14 → 误差 (10-14)= -4, (18-14)= +4 → 相对误差 4/14 ≈ 0.28 > 0.15 → 需要校正
        let mut polylines = vec![
            vec![[0.0, 0.0], [100.0, 0.5]],   // y ~ 0.25
            vec![[0.0, 10.0], [100.0, 10.3]], // y ~ 10.15
            vec![[0.0, 28.0], [100.0, 27.8]], // y ~ 27.9
        ];

        parallel_uniform_spacing(&mut polylines, 3.0, 0.15);

        // 检查间距现在均匀 → 应该大约 13.8 每个间距
        let y1 = (polylines[0][0][1] + polylines[0][1][1]) / 2.0;
        let y2 = (polylines[1][0][1] + polylines[1][1][1]) / 2.0;
        let y3 = (polylines[2][0][1] + polylines[2][1][1]) / 2.0;

        let d1 = y2 - y1;
        let d2 = y3 - y2;
        assert!((d1 - d2).abs() < 0.5);
    }

    #[test]
    fn test_find_peaks() {
        let mut hist = [0; 180];
        hist[0] = 10;
        hist[88] = 2;
        hist[89] = 8;
        hist[90] = 9;
        hist[91] = 7;
        hist[179] = 3;

        let peaks = find_peaks(&hist, 3);
        assert!(!peaks.is_empty());
        // 第一个峰值应该是 0 度
        assert_eq!(peaks[0].0, 0);
        // 第二个是 90 度
        assert_eq!(peaks[1].0, 90);
    }
}
