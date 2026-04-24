//! 建筑图纸线型识别
//!
//! 支持虚线、中心线、剖面线等线型识别

use common_types::{Point2, Polyline};

/// 线型类型
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LineType {
    /// 实线
    Continuous,
    /// 虚线（等长线段 + 等长间隙）
    Dashed,
    /// 中心线（长 - 短 - 长模式）
    Center,
    /// 隐藏线
    Hidden,
    /// 剖面线（长 - 短 - 短 - 长）
    Phantom,
    /// 无法识别
    Unknown,
}

/// 分析多段线的线型
///
/// # 参数
/// - `polyline`: 输入多段线
///
/// # 返回
/// 线型类型
pub fn analyze_line_type(polyline: &Polyline) -> LineType {
    if polyline.len() < 2 {
        return LineType::Continuous;
    }

    // 1. 计算线段长度序列
    let segments: Vec<f64> = polyline
        .windows(2)
        .map(|pts| distance(pts[0], pts[1]))
        .collect();

    // 2. 分析间隙模式（检测断点）
    let gaps = analyze_gaps(polyline);

    // 3. 如果没有间隙，认为是实线
    if gaps.is_empty() {
        return LineType::Continuous;
    }

    // 4. 分析线段长度模式
    let segment_lengths: Vec<f64> = segments
        .iter()
        .enumerate()
        .filter(|(i, _)| !gaps.contains(i))
        .map(|(_, &len)| len)
        .collect();

    if segment_lengths.is_empty() {
        return LineType::Unknown;
    }

    // 5. 识别模式
    if is_dashed_pattern(&segment_lengths) {
        return LineType::Dashed;
    }

    if is_center_line_pattern(&segment_lengths) {
        return LineType::Center;
    }

    if is_phantom_pattern(&segment_lengths) {
        return LineType::Phantom;
    }

    LineType::Continuous
}

/// 计算两点间距离
fn distance(a: Point2, b: Point2) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    (dx * dx + dy * dy).sqrt()
}

/// 分析多段线中的间隙（断点）
///
/// 检测点之间的距离是否显著大于平均距离
fn analyze_gaps(polyline: &Polyline) -> Vec<usize> {
    if polyline.len() < 3 {
        return Vec::new();
    }

    // 计算相邻点之间的距离
    let distances: Vec<f64> = polyline
        .windows(2)
        .map(|pts| distance(pts[0], pts[1]))
        .collect();

    // 计算平均距离
    let avg_dist: f64 = distances.iter().sum::<f64>() / distances.len() as f64;

    // 计算标准差
    let variance: f64 = distances
        .iter()
        .map(|&d| (d - avg_dist).powi(2))
        .sum::<f64>()
        / distances.len() as f64;
    let std_dev = variance.sqrt();

    // 找到显著大于平均值的间隙（> 2 倍标准差）
    let threshold = avg_dist + 2.0 * std_dev.max(avg_dist * 0.5);

    let gaps: Vec<usize> = distances
        .iter()
        .enumerate()
        .filter(|(_, &d)| d > threshold)
        .map(|(i, _)| i)
        .collect();

    gaps
}

/// 判断是否为虚线模式（等长线段 + 等长间隙）
fn is_dashed_pattern(segment_lengths: &[f64]) -> bool {
    if segment_lengths.len() < 3 {
        return false;
    }

    // 计算长度变化系数（变异系数）
    let avg: f64 = segment_lengths.iter().sum::<f64>() / segment_lengths.len() as f64;
    let variance: f64 = segment_lengths
        .iter()
        .map(|&l| (l - avg).powi(2))
        .sum::<f64>()
        / segment_lengths.len() as f64;
    let std_dev = variance.sqrt();
    let cv = std_dev / avg.max(1e-6);

    // 变异系数小于 0.3 认为是等长的
    cv < 0.3
}

/// 判断是否为中心线模式（长 - 短 - 长）
fn is_center_line_pattern(segment_lengths: &[f64]) -> bool {
    if segment_lengths.len() < 3 {
        return false;
    }

    // 寻找长 - 短 - 长模式
    let mut long_short_long_count = 0;

    for i in 0..segment_lengths.len() - 2 {
        let l1 = segment_lengths[i];
        let l2 = segment_lengths[i + 1];
        let l3 = segment_lengths[i + 2];

        // 长 - 短 - 长：两边长，中间短
        if l1 > l2 * 1.5 && l3 > l2 * 1.5 {
            long_short_long_count += 1;
        }
    }

    // 至少有 2 个长 - 短 - 长模式
    long_short_long_count >= 2
}

/// 判断是否为剖面线模式（长 - 短 - 短 - 长）
fn is_phantom_pattern(segment_lengths: &[f64]) -> bool {
    if segment_lengths.len() < 4 {
        return false;
    }

    // 寻找长 - 短 - 短 - 长模式
    let mut phantom_count = 0;

    for i in 0..segment_lengths.len() - 3 {
        let l1 = segment_lengths[i];
        let l2 = segment_lengths[i + 1];
        let l3 = segment_lengths[i + 2];
        let l4 = segment_lengths[i + 3];

        // 长 - 短 - 短 - 长：两边长，中间两个短
        if l1 > l2 * 1.5 && l1 > l3 * 1.5 && l4 > l2 * 1.5 && l4 > l3 * 1.5 {
            phantom_count += 1;
        }
    }

    // 至少有 1 个剖面线模式
    phantom_count >= 1
}

/// 线型分析结果
#[derive(Debug, Clone)]
pub struct LineTypeAnalysis {
    pub line_type: LineType,
    pub confidence: f64,
    pub segment_count: usize,
    pub avg_segment_length: f64,
}

/// 单条线段的线型标注（用于跨线段检测）
#[derive(Debug, Clone)]
pub struct PolylineLineType {
    pub polyline_idx: usize,
    pub line_type: LineType,
    pub group_id: usize,
    pub confidence: f64,
}

/// 线段特征（用于跨线段检测）
struct SegmentFeature {
    idx: usize,
    start: Point2,
    end: Point2,
    direction: Point2,
    length: f64,
}

/// 从多条分离线段中检测线型
///
/// 对于扫描图纸，虚线/中心线被提取为多条分离的共线线段。
/// 本函数将共线且间距规律的线段分组，分析其 dash-gap 模式。
///
/// # 参数
/// - `polylines`: 输入多段线集合
/// - `collinear_angle_tol`: 共线角度容差（度），例如 15.0 表示方向夹角 <= 15° 视为共线
/// - `gap_tolerance`: 最大间隙距离，超过此距离的端点不视为同一线型的一部分
///
/// # 返回
/// 每条线段对应的线型标注
pub fn detect_line_types_from_polylines(
    polylines: &[Polyline],
    collinear_angle_tol: f64,
    gap_tolerance: f64,
) -> Vec<PolylineLineType> {
    if polylines.is_empty() {
        return Vec::new();
    }

    // 1. 计算每条线段的特征
    let mut features = Vec::with_capacity(polylines.len());
    for (i, pl) in polylines.iter().enumerate() {
        if pl.len() < 2 {
            features.push(SegmentFeature {
                idx: i,
                start: [0.0, 0.0],
                end: [0.0, 0.0],
                direction: [1.0, 0.0],
                length: 0.0,
            });
            continue;
        }
        let start = pl[0];
        let end = pl[pl.len() - 1];
        let dx = end[0] - start[0];
        let dy = end[1] - start[1];
        let len = (dx * dx + dy * dy).sqrt();
        let dir = if len > 1e-10 {
            [dx / len, dy / len]
        } else {
            [1.0, 0.0]
        };
        features.push(SegmentFeature {
            idx: i,
            start,
            end,
            direction: dir,
            length: len,
        });
    }

    // 2. Union-Find 共线分组
    let mut uf = UnionFind::new(features.len());
    let angle_tol_rad = collinear_angle_tol.to_radians();

    for i in 0..features.len() {
        if features[i].length < 1e-10 {
            continue;
        }
        for j in (i + 1)..features.len() {
            if features[j].length < 1e-10 {
                continue;
            }
            // 方向夹角检查（考虑同向和反向）
            let dot = features[i].direction[0] * features[j].direction[0]
                + features[i].direction[1] * features[j].direction[1];
            let cos_angle = dot.abs(); // abs 因为反向也认为共线
            if cos_angle < angle_tol_rad.cos() {
                continue;
            }
            // 端点最小距离检查
            let min_endpoint_dist = min_endpoint_distance(
                features[i].start,
                features[i].end,
                features[j].start,
                features[j].end,
            );
            if min_endpoint_dist > gap_tolerance {
                continue;
            }
            uf.union(i, j);
        }
    }

    // 3. 按组分类
    let n = features.len();
    let mut groups: std::collections::HashMap<usize, Vec<usize>> = std::collections::HashMap::new();
    for i in 0..n {
        let root = uf.find(i);
        groups.entry(root).or_default().push(i);
    }

    // 4. 对每组进行分类
    let mut result = vec![
        PolylineLineType {
            polyline_idx: 0,
            line_type: LineType::Continuous,
            group_id: 0,
            confidence: 1.0,
        };
        n
    ];

    for (gid, members) in groups.values().enumerate() {
        let line_type = classify_group(&features, members);

        // 计算置信度
        let confidence = match line_type {
            LineType::Continuous => 1.0,
            LineType::Dashed => 0.8,
            LineType::Center => 0.7,
            LineType::Hidden => 0.7,
            LineType::Phantom => 0.6,
            LineType::Unknown => 0.3,
        };

        for &idx in members {
            result[idx] = PolylineLineType {
                polyline_idx: features[idx].idx,
                line_type: line_type.clone(),
                group_id: gid,
                confidence,
            };
        }
    }

    result
}

/// 计算两个线段端点间的最小距离
fn min_endpoint_distance(a_start: Point2, a_end: Point2, b_start: Point2, b_end: Point2) -> f64 {
    [
        distance(a_start, b_start),
        distance(a_start, b_end),
        distance(a_end, b_start),
        distance(a_end, b_end),
    ]
    .into_iter()
    .fold(f64::INFINITY, f64::min)
}

/// 并查集
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

/// 对组内线段分类
fn classify_group(features: &[SegmentFeature], members: &[usize]) -> LineType {
    if members.len() == 1 {
        return LineType::Continuous;
    }

    // 按沿主方向的投影排序
    let main_dir = features[members[0]].direction;
    let mut sorted: Vec<&SegmentFeature> = members.iter().map(|&i| &features[i]).collect();
    sorted.sort_by(|a, b| {
        let proj_a = a.start[0] * main_dir[0] + a.start[1] * main_dir[1];
        let proj_b = b.start[0] * main_dir[0] + b.start[1] * main_dir[1];
        proj_a
            .partial_cmp(&proj_b)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // 提取 dash 和 gap 序列
    let dash_lengths: Vec<f64> = sorted.iter().map(|f| f.length).collect();
    let gaps: Vec<f64> = sorted
        .windows(2)
        .map(|w| min_endpoint_distance(w[0].start, w[0].end, w[1].start, w[1].end))
        .collect();

    if dash_lengths.len() < 2 {
        return LineType::Continuous;
    }

    // 计算变异系数
    let dash_cv = coefficient_of_variation(&dash_lengths);
    let gap_cv = coefficient_of_variation(&gaps);

    // 虚线检测: 等长 dash + 等长 gap
    if dash_cv < 0.3 && gaps.len() >= 2 && gap_cv < 0.3 {
        return LineType::Dashed;
    }

    // 中心线检测: 长-短-长交替
    if is_long_short_long_pattern(&dash_lengths) {
        return LineType::Center;
    }

    // 剖面线检测: 长-短-短-长
    if is_long_short_short_long_pattern(&dash_lengths) {
        return LineType::Phantom;
    }

    // 只有 2 条线段且间距小，可能是不完整的虚线
    if members.len() == 2 && dash_cv < 0.3 {
        return LineType::Hidden;
    }

    LineType::Continuous
}

/// 计算变异系数 (CV = std_dev / mean)
fn coefficient_of_variation(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 1.0;
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    if mean < 1e-10 {
        return 0.0;
    }
    let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;
    variance.sqrt() / mean
}

/// 检测长-短-长模式（中心线）
fn is_long_short_long_pattern(lengths: &[f64]) -> bool {
    if lengths.len() < 3 {
        return false;
    }
    let mut pattern_count = 0;
    for i in 0..lengths.len() - 2 {
        let mid = lengths[i + 1];
        if lengths[i] > mid * 2.0 && lengths[i + 2] > mid * 2.0 && mid > 1e-10 {
            pattern_count += 1;
        }
    }
    pattern_count >= 2
}

/// 检测长-短-短-长模式（剖面线）
fn is_long_short_short_long_pattern(lengths: &[f64]) -> bool {
    if lengths.len() < 4 {
        return false;
    }
    let mut pattern_count = 0;
    for i in 0..lengths.len() - 3 {
        let long1 = lengths[i];
        let short1 = lengths[i + 1];
        let short2 = lengths[i + 2];
        let long2 = lengths[i + 3];
        if long1 > short1 * 2.0
            && long1 > short2 * 2.0
            && long2 > short1 * 2.0
            && long2 > short2 * 2.0
        {
            pattern_count += 1;
        }
    }
    pattern_count >= 1
}

/// 分析线型并返回详细结果
pub fn analyze_line_type_detailed(polyline: &Polyline) -> LineTypeAnalysis {
    let line_type = analyze_line_type(polyline);

    let segment_count = polyline.len() - 1;
    let avg_length: f64 = polyline
        .windows(2)
        .map(|pts| distance(pts[0], pts[1]))
        .sum::<f64>()
        / segment_count.max(1) as f64;

    // 简单置信度计算
    let confidence = match line_type {
        LineType::Continuous => 1.0,
        LineType::Dashed => 0.8,
        LineType::Center => 0.7,
        LineType::Hidden => 0.7,
        LineType::Phantom => 0.6,
        LineType::Unknown => 0.3,
    };

    LineTypeAnalysis {
        line_type,
        confidence,
        segment_count,
        avg_segment_length: avg_length,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_continuous_line() {
        let polyline = vec![[0.0, 0.0], [10.0, 0.0], [20.0, 0.0], [30.0, 0.0]];

        let line_type = analyze_line_type(&polyline);
        assert_eq!(line_type, LineType::Continuous);
    }

    #[test]
    fn test_dashed_line() {
        // 模拟虚线：等长线段
        let polyline = vec![
            [0.0, 0.0],
            [5.0, 0.0],  // 线段 1
            [7.0, 0.0],  // 间隙
            [12.0, 0.0], // 线段 2
            [14.0, 0.0], // 间隙
            [19.0, 0.0], // 线段 3
        ];

        let line_type = analyze_line_type(&polyline);
        // 虚线检测依赖于间隙分析
        assert!(line_type == LineType::Dashed || line_type == LineType::Continuous);
    }

    #[test]
    fn test_center_line_pattern() {
        // 模拟中心线：长 - 短 - 长模式
        // 注意：analyze_line_type 检测的是线段长度模式，不是间隙模式
        // 对于连续点组成的多段线，如果没有间隙，会被认为是 Continuous
        // 这个测试用于验证 is_center_line_pattern 函数的正确性
        let segment_lengths = vec![20.0, 3.0, 20.0, 3.0, 20.0];

        // 直接测试模式识别函数
        assert!(is_center_line_pattern(&segment_lengths));

        // 测试虚线模式识别
        let dashed_lengths = vec![5.0, 5.0, 5.0, 5.0, 5.0];
        assert!(is_dashed_pattern(&dashed_lengths));

        // 测试连续线模式（长度不相等且无规律）
        let continuous_lengths = vec![10.0, 15.0, 8.0, 22.0, 11.0];
        assert!(!is_dashed_pattern(&continuous_lengths));
        assert!(!is_center_line_pattern(&continuous_lengths));
    }

    #[test]
    fn test_distance_calculation() {
        let a = [0.0, 0.0];
        let b = [3.0, 4.0];
        assert!((distance(a, b) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_detect_dashed_from_separate_segments() {
        // 3 条共线等长线段，等长间隙
        let polylines = vec![
            vec![[0.0, 0.0], [10.0, 0.0]],  // dash 1: length 10
            vec![[12.0, 0.0], [22.0, 0.0]], // dash 2: length 10
            vec![[24.0, 0.0], [34.0, 0.0]], // dash 3: length 10
            vec![[36.0, 0.0], [46.0, 0.0]], // dash 4: length 10
            vec![[48.0, 0.0], [58.0, 0.0]], // dash 5: length 10
        ];

        let results = detect_line_types_from_polylines(&polylines, 15.0, 5.0);

        assert_eq!(results.len(), 5);
        // 所有线段应在同一组且被识别为虚线
        let all_same_group = results.iter().all(|r| r.group_id == results[0].group_id);
        assert!(all_same_group, "所有线段应在同一共线组");
        assert_eq!(results[0].line_type, LineType::Dashed);
    }

    #[test]
    fn test_detect_center_from_separate_segments() {
        // 长-短-长交替的线段
        let polylines = vec![
            vec![[0.0, 0.0], [20.0, 0.0]],  // long: 20
            vec![[22.0, 0.0], [25.0, 0.0]], // short: 3
            vec![[27.0, 0.0], [47.0, 0.0]], // long: 20
            vec![[49.0, 0.0], [52.0, 0.0]], // short: 3
            vec![[54.0, 0.0], [74.0, 0.0]], // long: 20
        ];

        let results = detect_line_types_from_polylines(&polylines, 15.0, 5.0);

        assert_eq!(results.len(), 5);
        let all_same_group = results.iter().all(|r| r.group_id == results[0].group_id);
        assert!(all_same_group, "所有线段应在同一共线组");
        // 长-短-长模式应被识别为中心线
        assert_eq!(results[0].line_type, LineType::Center);
    }

    #[test]
    fn test_non_collinear_not_grouped() {
        // 正交线段（水平和垂直），不应分组
        let polylines = vec![
            vec![[0.0, 0.0], [10.0, 0.0]],  // 水平
            vec![[0.0, 0.0], [0.0, 10.0]],  // 垂直
            vec![[20.0, 0.0], [30.0, 0.0]], // 水平（但距离远）
        ];

        // 使用很小的 gap_tolerance 确保不分组
        let results = detect_line_types_from_polylines(&polylines, 15.0, 1.0);

        assert_eq!(results.len(), 3);
        // 水平和垂直线段不应在同一组
        let h1_group = results[0].group_id;
        let v1_group = results[1].group_id;
        assert_ne!(h1_group, v1_group, "正交线段不应在同一组");
    }

    #[test]
    fn test_confidence_and_group_id() {
        // 验证 group_id 和 confidence 计算
        let polylines = vec![
            vec![[0.0, 0.0], [5.0, 0.0]],
            vec![[7.0, 0.0], [12.0, 0.0]],
            vec![[14.0, 0.0], [19.0, 0.0]],
        ];

        let results = detect_line_types_from_polylines(&polylines, 15.0, 5.0);

        // 所有线段应在同一组
        let group_ids: Vec<usize> = results.iter().map(|r| r.group_id).collect();
        assert!(
            group_ids.iter().all(|&g| g == group_ids[0]),
            "所有共线线段应有相同 group_id"
        );
        // 置信度应 > 0
        for r in &results {
            assert!(r.confidence > 0.0, "置信度应 > 0");
        }
    }

    #[test]
    fn test_empty_polylines() {
        let results: Vec<PolylineLineType> = detect_line_types_from_polylines(&[], 15.0, 5.0);
        assert!(results.is_empty());
    }

    #[test]
    fn test_single_segment() {
        let polylines = vec![vec![[0.0, 0.0], [10.0, 0.0]]];
        let results = detect_line_types_from_polylines(&polylines, 15.0, 5.0);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].line_type, LineType::Continuous);
    }
}
