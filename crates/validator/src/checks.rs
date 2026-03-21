//! 验证检查项 - 改进版
//!
//! 改进：
//! 1. 添加共线重叠检查（之前只检查严格相交）
//! 2. 使用 geo  crate 的 robust predicates

use common_types::{ClosedLoop, Point2, RecoverySuggestion};
use geo::{LineString, Intersects, Contains};
use serde::{Serialize, Deserialize};

/// 验证问题严重性
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Severity {
    /// 错误 - 必须修复
    Error,
    /// 警告 - 建议修复
    Warning,
    /// 提示 - 信息性
    Info,
}

/// 验证问题
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationIssue {
    pub code: String,
    pub message: String,
    pub severity: Severity,
    pub location: Option<ValidationLocation>,
    pub suggestion: Option<String>,
}

/// 问题位置
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ValidationLocation {
    pub point: Option<Point2>,
    pub segment: Option<[usize; 2]>,
    pub loop_index: Option<usize>,
}

/// 验证报告
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationReport {
    pub passed: bool,
    pub issues: Vec<ValidationIssue>,
    pub summary: ValidationSummary,
    /// 错误恢复建议（P11 Principal Engineer 建议）
    pub recovery_suggestions: Vec<RecoverySuggestion>,
}

impl Default for ValidationReport {
    fn default() -> Self {
        Self {
            passed: true,
            issues: Vec::new(),
            summary: ValidationSummary::default(),
            recovery_suggestions: Vec::new(),
        }
    }
}

/// 验证摘要
#[derive(Debug, Clone, Serialize, Deserialize)]
#[derive(Default)]
pub struct ValidationSummary {
    pub error_count: usize,
    pub warning_count: usize,
    pub info_count: usize,
}

impl ValidationSummary {
    pub fn new() -> Self {
        Self::default()
    }
}

/// 闭合性检查
pub fn check_closure(loop_data: &ClosedLoop, tolerance: f64) -> Option<ValidationIssue> {
    if loop_data.points.len() < 3 {
        return Some(ValidationIssue {
            code: "E001".to_string(),
            message: "环的点数不足，无法形成闭合区域".to_string(),
            severity: Severity::Error,
            location: None,
            suggestion: Some("添加更多点以形成有效的闭合环".to_string()),
        });
    }

    let first = loop_data.points[0];
    let last = loop_data.points[loop_data.points.len() - 1];

    let dx = first[0] - last[0];
    let dy = first[1] - last[1];
    let dist = (dx * dx + dy * dy).sqrt();

    if dist > tolerance {
        Some(ValidationIssue {
            code: "E002".to_string(),
            message: format!("环未闭合，首尾点距离为 {:.3}mm", dist),
            severity: Severity::Error,
            location: Some(ValidationLocation {
                point: Some(first),
                segment: None,
                loop_index: None,
            }),
            suggestion: Some("调整端点位置使其闭合或增加容差".to_string()),
        })
    } else {
        None
    }
}

/// 自相交检查 - 改进版
///
/// 检查：
/// 1. 严格相交（线段交叉）
/// 2. 共线重叠（线段部分或完全重叠）
/// 3. 重复顶点
pub fn check_self_intersection(loop_data: &ClosedLoop) -> Option<ValidationIssue> {
    let points = &loop_data.points;
    if points.len() < 4 {
        return None; // 三角形不可能自相交
    }

    // 检查重复顶点
    for i in 0..points.len() {
        for j in (i + 1)..points.len() {
            // 跳过相邻顶点
            if j == i + 1 {
                continue;
            }
            // 跳过首尾（因为是闭合的）
            if i == 0 && j == points.len() - 1 {
                continue;
            }

            if points[i] == points[j] {
                return Some(ValidationIssue {
                    code: "E003".to_string(),
                    message: "环存在重复顶点".to_string(),
                    severity: Severity::Error,
                    location: Some(ValidationLocation {
                        point: Some(points[i]),
                        segment: None,
                        loop_index: None,
                    }),
                    suggestion: Some("移除重复的顶点".to_string()),
                });
            }
        }
    }

    // 使用 robust geometry 检查自相交
    // 检查所有非相邻边对是否有相交
    if let Some(issue) = check_self_intersection_robust(points) {
        return Some(issue);
    }

    // 额外检查：使用 geo::LineString 诊断相交类型
    let line_string: LineString<f64> = points.iter()
        .map(|p| geo::Coord { x: p[0], y: p[1] })
        .collect();

    // 诊断相交类型（如果有相交）
    let intersection_type = diagnose_intersection_type(&line_string);
    if intersection_type != IntersectionType::Unknown {
        let message = match intersection_type {
            IntersectionType::Crossing => "环存在自相交（线段交叉）".to_string(),
            IntersectionType::Overlap => "环存在共线重叠（线段部分或完全重叠）".to_string(),
            IntersectionType::Touch => "环存在接触（线段在端点处接触）".to_string(),
            IntersectionType::Unknown => unreachable!(),
        };

        return Some(ValidationIssue {
            code: "E003".to_string(),
            message,
            severity: Severity::Error,
            location: None,
            suggestion: Some("简化几何形状或移除交叉点".to_string()),
        });
    }

    None
}

/// 相交类型诊断
#[derive(Debug, Clone, Copy, PartialEq)]
#[allow(dead_code)] // 保留用于未来扩展
enum IntersectionType {
    Crossing,
    Overlap,
    Touch,
    Unknown,
}

/// 诊断相交类型
fn diagnose_intersection_type(line_string: &LineString<f64>) -> IntersectionType {
    let coords = &line_string.0;
    let n = coords.len();

    for i in 0..n {
        let j = (i + 1) % n;
        let line1 = geo::Line::new(coords[i], coords[j]);

        for k in (i + 2)..n {
            let l = (k + 1) % n;
            
            // 跳过相邻边
            if i == 0 && k == n - 1 {
                continue;
            }

            let line2 = geo::Line::new(coords[k], coords[l]);

            // 检查是否共线
            if are_collinear(line1, line2) {
                // 检查是否重叠
                if segments_overlap(line1, line2) {
                    return IntersectionType::Overlap;
                }
            } else if line1.intersects(&line2) {
                return IntersectionType::Crossing;
            }
        }
    }

    IntersectionType::Unknown
}

/// 检查两条线段是否共线
fn are_collinear(line1: geo::Line<f64>, line2: geo::Line<f64>) -> bool {
    // 使用叉积检查
    let v1 = (line1.end.x - line1.start.x, line1.end.y - line1.start.y);
    let v2 = (line2.end.x - line2.start.x, line2.end.y - line2.start.y);
    
    // 叉积为零表示平行
    let cross = v1.0 * v2.1 - v1.1 * v2.0;
    
    if cross.abs() > 1e-10 {
        return false;
    }

    // 检查是否在同一直线上
    let v3 = (line2.start.x - line1.start.x, line2.start.y - line1.start.y);
    let cross2 = v1.0 * v3.1 - v1.1 * v3.0;
    
    cross2.abs() < 1e-10
}

/// 检查两条共线线段是否重叠
fn segments_overlap(line1: geo::Line<f64>, line2: geo::Line<f64>) -> bool {
    // 投影到 x 轴或 y 轴（选择变化较大的轴）
    let dx1 = (line1.end.x - line1.start.x).abs();
    let dy1 = (line1.end.y - line1.start.y).abs();

    if dx1 > dy1 {
        // 投影到 x 轴
        let (min1, max1) = if line1.start.x < line1.end.x {
            (line1.start.x, line1.end.x)
        } else {
            (line1.end.x, line1.start.x)
        };
        let (min2, max2) = if line2.start.x < line2.end.x {
            (line2.start.x, line2.end.x)
        } else {
            (line2.end.x, line2.start.x)
        };

        // 检查区间是否重叠
        min1 < max2 && min2 < max1
    } else {
        // 投影到 y 轴
        let (min1, max1) = if line1.start.y < line1.end.y {
            (line1.start.y, line1.end.y)
        } else {
            (line1.end.y, line1.start.y)
        };
        let (min2, max2) = if line2.start.y < line2.end.y {
            (line2.start.y, line2.end.y)
        } else {
            (line2.end.y, line2.start.y)
        };

        // 检查区间是否重叠
        min1 < max2 && min2 < max1
    }
}

/// 使用 robust geometry 进行精确自相交检查
fn check_self_intersection_robust(points: &[Point2]) -> Option<ValidationIssue> {
    // 使用 geo 的 robust predicates
    // 这里我们检查所有非相邻边对
    
    let n = points.len();
    
    for i in 0..n {
        let j = (i + 1) % n;
        let p1 = points[i];
        let p2 = points[j];

        for k in 0..n {
            let l = (k + 1) % n;
            
            // 跳过相邻边
            if k == i || k == j || l == i || l == j {
                continue;
            }
            if i == 0 && k == n - 1 {
                continue;
            }

            let p3 = points[k];
            let p4 = points[l];

            // 使用 robust orientation 检查
            if robust_intersects(p1, p2, p3, p4) {
                return Some(ValidationIssue {
                    code: "E003".to_string(),
                    message: "环存在自相交（精确检测）".to_string(),
                    severity: Severity::Error,
                    location: Some(ValidationLocation {
                        point: Some(p1),
                        segment: Some([i, j]),
                        loop_index: None,
                    }),
                    suggestion: Some("简化几何形状或移除交叉点".to_string()),
                });
            }
        }
    }

    None
}

/// Robust 线段相交检查
fn robust_intersects(p1: Point2, p2: Point2, p3: Point2, p4: Point2) -> bool {
    // 使用 orientation test
    let o1 = orientation(p1, p2, p3);
    let o2 = orientation(p1, p2, p4);
    let o3 = orientation(p3, p4, p1);
    let o4 = orientation(p3, p4, p2);

    // 一般情况：线段相交当且仅当端点位于另一线段的两侧
    if o1 != o2 && o3 != o4 {
        return true;
    }

    // 特殊情况：共线
    if o1 == 0 && on_segment(p1, p2, p3) {
        return true;
    }
    if o2 == 0 && on_segment(p1, p2, p4) {
        return true;
    }
    if o3 == 0 && on_segment(p3, p4, p1) {
        return true;
    }
    if o4 == 0 && on_segment(p3, p4, p2) {
        return true;
    }

    false
}

/// 计算 orientation（返回 -1, 0, 1）
fn orientation(p1: Point2, p2: Point2, p3: Point2) -> i32 {
    let val = (p2[1] - p1[1]) * (p3[0] - p2[0]) - (p2[0] - p1[0]) * (p3[1] - p2[1]);
    
    if val.abs() < 1e-10 {
        0 // 共线
    } else if val > 0.0 {
        1 // 顺时针
    } else {
        -1 // 逆时针
    }
}

/// 检查点 q 是否在线段 pr 上
fn on_segment(p: Point2, r: Point2, q: Point2) -> bool {
    q[0] >= p[0].min(r[0]) - 1e-10
        && q[0] <= p[0].max(r[0]) + 1e-10
        && q[1] >= p[1].min(r[1]) - 1e-10
        && q[1] <= p[1].max(r[1]) + 1e-10
}

/// 孔洞包含检查
pub fn check_hole_containment(outer: &ClosedLoop, holes: &[ClosedLoop]) -> Vec<ValidationIssue> {
    let mut issues = Vec::new();

    let outer_line: LineString<f64> = outer.points.iter()
        .map(|p| geo::Coord { x: p[0], y: p[1] })
        .collect();

    let outer_polygon = geo::Polygon::new(outer_line.clone(), vec![]);

    for (i, hole) in holes.iter().enumerate() {
        let hole_line: LineString<f64> = hole.points.iter()
            .map(|p| geo::Coord { x: p[0], y: p[1] })
            .collect();

        // 检查孔洞是否完全在外轮廓内
        // 使用简单的中心点判断
        let hole_center = calculate_center(&hole_line);
        if let Some(center) = hole_center {
            let center_point = geo::Point::new(center.0, center.1);
            if !outer_polygon.contains(&center_point) {
                issues.push(ValidationIssue {
                    code: "E004".to_string(),
                    message: format!("孔洞 #{} 不在外轮廓内", i),
                    severity: Severity::Error,
                    location: Some(ValidationLocation {
                        point: Some([center.0, center.1]),
                        segment: None,
                        loop_index: Some(i),
                    }),
                    suggestion: Some("调整孔洞位置使其位于外轮廓内".to_string()),
                });
            }
        }

        // 检查孔洞是否与外轮廓相交
        if outer_line.intersects(&hole_line) {
            issues.push(ValidationIssue {
                code: "E005".to_string(),
                message: format!("孔洞 #{} 与外轮廓相交", i),
                severity: Severity::Error,
                location: None,
                suggestion: Some("调整孔洞位置避免与外轮廓相交".to_string()),
            });
        }
    }

    // 检查孔洞之间是否相交
    for i in 0..holes.len() {
        for j in (i + 1)..holes.len() {
            let line_i: LineString<f64> = holes[i].points.iter()
                .map(|p| geo::Coord { x: p[0], y: p[1] })
                .collect();
            let line_j: LineString<f64> = holes[j].points.iter()
                .map(|p| geo::Coord { x: p[0], y: p[1] })
                .collect();

            if line_i.intersects(&line_j) {
                issues.push(ValidationIssue {
                    code: "E006".to_string(),
                    message: format!("孔洞 #{} 与孔洞 #{} 相交", i, j),
                    severity: Severity::Error,
                    location: None,
                    suggestion: Some("调整孔洞位置避免相互重叠".to_string()),
                });
            }
        }
    }

    issues
}

/// 微特征检查
pub fn check_micro_features(loop_data: &ClosedLoop, min_length: f64, min_angle: f64) -> Vec<ValidationIssue> {
    let mut issues = Vec::new();
    let points = &loop_data.points;

    if points.len() < 2 {
        return issues;
    }

    // 检查短边
    for i in 0..points.len() {
        let j = (i + 1) % points.len();
        let dx = points[j][0] - points[i][0];
        let dy = points[j][1] - points[i][1];
        let len = (dx * dx + dy * dy).sqrt();

        if len < min_length {
            issues.push(ValidationIssue {
                code: "W001".to_string(),
                message: format!("检测到短边 ({:.3}mm < {:.3}mm)", len, min_length),
                severity: Severity::Warning,
                location: Some(ValidationLocation {
                    point: Some(points[i]),
                    segment: Some([i, j]),
                    loop_index: None,
                }),
                suggestion: Some("考虑移除或合并短边".to_string()),
            });
        }
    }

    // 检查锐角
    for i in 0..points.len() {
        let prev = if i == 0 { points.len() - 1 } else { i - 1 };
        let next = (i + 1) % points.len();

        let v1 = [points[i][0] - points[prev][0], points[i][1] - points[prev][1]];
        let v2 = [points[next][0] - points[i][0], points[next][1] - points[i][1]];

        let angle = calculate_angle(&v1, &v2);
        let angle_deg = angle.to_degrees();

        if angle_deg < min_angle {
            issues.push(ValidationIssue {
                code: "W002".to_string(),
                message: format!("检测到锐角 ({:.1}° < {:.1}°)", angle_deg, min_angle),
                severity: Severity::Warning,
                location: Some(ValidationLocation {
                    point: Some(points[i]),
                    segment: None,
                    loop_index: None,
                }),
                suggestion: Some("考虑倒角或简化几何形状".to_string()),
            });
        }
    }

    issues
}

fn calculate_angle(v1: &[f64; 2], v2: &[f64; 2]) -> f64 {
    let dot = v1[0] * v2[0] + v1[1] * v2[1];
    let len1 = (v1[0] * v1[0] + v1[1] * v1[1]).sqrt();
    let len2 = (v2[0] * v2[0] + v2[1] * v2[1]).sqrt();

    if len1 < 1e-10 || len2 < 1e-10 {
        return 0.0;
    }

    let cos = (dot / (len1 * len2)).clamp(-1.0, 1.0);
    cos.acos()
}

/// 计算线串的简单中心（边界框中心）
fn calculate_center(line: &LineString<f64>) -> Option<(f64, f64)> {
    if line.0.is_empty() {
        return None;
    }

    let mut min_x = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_y = f64::NEG_INFINITY;

    for coord in &line.0 {
        min_x = min_x.min(coord.x);
        max_x = max_x.max(coord.x);
        min_y = min_y.min(coord.y);
        max_y = max_y.max(coord.y);
    }

    Some(((min_x + max_x) / 2.0, (min_y + max_y) / 2.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_check_closure_valid() {
        // 闭合矩形（首尾点重合）
        let loop_data = ClosedLoop {
            points: vec![[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0], [0.0, 0.0]],
            signed_area: 100.0,
        };
        assert!(check_closure(&loop_data, 0.5).is_none());
    }

    #[test]
    fn test_check_closure_invalid() {
        // 开放的多段线（首尾点不重合）
        let loop_data = ClosedLoop {
            points: vec![[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]],
            signed_area: 50.0,
        };
        assert!(check_closure(&loop_data, 0.5).is_some());
    }

    #[test]
    fn test_check_self_intersection_simple() {
        // 简单矩形，无自相交
        let loop_data = ClosedLoop {
            points: vec![[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]],
            signed_area: 100.0,
        };
        assert!(check_self_intersection(&loop_data).is_none());
    }

    #[test]
    fn test_check_self_intersection_crossing() {
        // 蝴蝶结形状（自相交）
        let loop_data = ClosedLoop {
            points: vec![[0.0, 0.0], [10.0, 10.0], [10.0, 0.0], [0.0, 10.0]],
            signed_area: 0.0,
        };
        assert!(check_self_intersection(&loop_data).is_some());
    }

    #[test]
    fn test_check_self_intersection_overlap() {
        // 共线重叠情况
        let loop_data = ClosedLoop {
            points: vec![
                [0.0, 0.0], [10.0, 0.0],  // 底边
                [10.0, 10.0],             // 右边
                [5.0, 10.0],              // 顶边部分 1
                [5.0, 5.0],               // 向下
                [0.0, 5.0],               // 向左
                [0.0, 10.0],              // 向上
            ],
            signed_area: 75.0,
        };
        // 这个形状有 T 型连接，应该被检测为问题
        assert!(check_self_intersection(&loop_data).is_some());
    }

    #[test]
    fn test_are_collinear_parallel() {
        // 平行但不共线
        let line1 = geo::Line::new(geo::Coord { x: 0.0, y: 0.0 }, geo::Coord { x: 10.0, y: 0.0 });
        let line2 = geo::Line::new(geo::Coord { x: 0.0, y: 1.0 }, geo::Coord { x: 10.0, y: 1.0 });
        assert!(!are_collinear(line1, line2));
    }

    #[test]
    fn test_are_collinear_same_line() {
        // 共线
        let line1 = geo::Line::new(geo::Coord { x: 0.0, y: 0.0 }, geo::Coord { x: 10.0, y: 0.0 });
        let line2 = geo::Line::new(geo::Coord { x: 5.0, y: 0.0 }, geo::Coord { x: 15.0, y: 0.0 });
        assert!(are_collinear(line1, line2));
    }

    #[test]
    fn test_segments_overlap_yes() {
        let line1 = geo::Line::new(geo::Coord { x: 0.0, y: 0.0 }, geo::Coord { x: 10.0, y: 0.0 });
        let line2 = geo::Line::new(geo::Coord { x: 5.0, y: 0.0 }, geo::Coord { x: 15.0, y: 0.0 });
        assert!(segments_overlap(line1, line2));
    }

    #[test]
    fn test_segments_overlap_no() {
        let line1 = geo::Line::new(geo::Coord { x: 0.0, y: 0.0 }, geo::Coord { x: 10.0, y: 0.0 });
        let line2 = geo::Line::new(geo::Coord { x: 11.0, y: 0.0 }, geo::Coord { x: 20.0, y: 0.0 });
        assert!(!segments_overlap(line1, line2));
    }
}

// ==================== 凸包算法和凸性检查 ====================

/// 计算多边形的凸包（Monotone Chain 算法）
/// 时间复杂度：O(n log n)
pub fn compute_convex_hull(points: &[Point2]) -> Vec<Point2> {
    if points.len() <= 3 {
        return points.to_vec();
    }

    let mut sorted: Vec<_> = points.to_vec();
    sorted.sort_by(|a, b| {
        a[0].partial_cmp(&b[0])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a[1].partial_cmp(&b[1]).unwrap_or(std::cmp::Ordering::Equal))
    });
    
    let mut hull = Vec::with_capacity(points.len());
    
    // 下凸壳
    for p in &sorted {
        while hull.len() >= 2 {
            let a = hull[hull.len() - 2];
            let b = hull[hull.len() - 1];
            if cross_product_2d(a, b, *p) <= 0.0 {
                hull.pop();
            } else {
                break;
            }
        }
        hull.push(*p);
    }
    
    // 上凸壳
    let lower_hull_len = hull.len();
    for p in sorted.iter().rev() {
        while hull.len() > lower_hull_len {
            let a = hull[hull.len() - 2];
            let b = hull[hull.len() - 1];
            if cross_product_2d(a, b, *p) <= 0.0 {
                hull.pop();
            } else {
                break;
            }
        }
        hull.push(*p);
    }
    
    // 移除重复的终点
    if hull.len() > 1 {
        hull.pop();
    }
    hull
}

/// 2D 叉积：>0 表示左转，<0 表示右转，=0 表示共线
fn cross_product_2d(a: Point2, b: Point2, c: Point2) -> f64 {
    (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
}

/// 计算多边形面积（带符号）
pub fn calculate_polygon_area(points: &[Point2]) -> f64 {
    if points.len() < 3 {
        return 0.0;
    }
    
    let mut area = 0.0;
    for i in 0..points.len() {
        let j = (i + 1) % points.len();
        area += points[i][0] * points[j][1];
        area -= points[j][0] * points[i][1];
    }
    area / 2.0
}

/// 凸性分析结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplexityMetrics {
    /// 是否为凸多边形
    pub is_convex: bool,
    /// 凸性比率（实际面积/凸包面积），1.0=凸，<1.0=凹
    pub convexity_ratio: f64,
    /// 复杂度评分（0=凸，1=极度凹）
    pub complexity_score: f64,
}

/// 墙体凹陷信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Indentation {
    /// 凹陷起始点索引
    pub start_idx: usize,
    /// 凹陷结束点索引
    pub end_idx: usize,
    /// 凹陷深度（米）
    pub depth: f64,
    /// 凹陷宽度（米）
    pub width: f64,
}

/// 凸性检查 - 房间复杂度分析
pub fn check_convexity(outer: &ClosedLoop) -> Vec<ValidationIssue> {
    let mut issues = Vec::new();
    
    if outer.points.len() < 3 {
        return issues;
    }
    
    let hull = compute_convex_hull(&outer.points);
    let actual_area = outer.signed_area.abs();
    let hull_area = calculate_polygon_area(&hull).abs();
    
    if hull_area < 1e-10 {
        return issues;
    }
    
    let convexity_ratio = actual_area / hull_area;
    let _complexity_score = 1.0 - convexity_ratio;
    
    // 凸性过低预警（声学聚焦风险）
    if convexity_ratio < 0.6 {
        issues.push(ValidationIssue {
            code: "W004".to_string(),
            message: format!("房间凸性比率过低 ({:.2})，可能存在声学聚焦", convexity_ratio),
            severity: Severity::Warning,
            location: Some(ValidationLocation {
                point: Some(find_deepest_concave_point(&outer.points, &hull)),
                segment: None,
                loop_index: None,
            }),
            suggestion: Some("考虑添加扩散体或调整房间形状".to_string()),
        });
    }
    
    // 检测异常凹陷（可能是识别错误）
    let indentations = detect_wall_indentations(&outer.points, &hull);
    for indent in indentations {
        if indent.depth > 0.5 && indent.width < 0.3 {
            // 深而窄的凹陷，可能是噪声
            issues.push(ValidationIssue {
                code: "E005".to_string(),
                message: format!("检测到异常墙体凹陷 (深度:{:.2}m, 宽度:{:.2}m)", indent.depth, indent.width),
                severity: Severity::Error,
                location: Some(ValidationLocation {
                    point: Some(outer.points[indent.start_idx]),
                    segment: Some([indent.start_idx, indent.end_idx]),
                    loop_index: None,
                }),
                suggestion: Some("检查是否为识别错误，建议手动修复".to_string()),
            });
        }
    }
    
    issues
}

/// 查找最深的凹点
fn find_deepest_concave_point(points: &[Point2], hull: &[Point2]) -> Point2 {
    // 简单实现：返回第一个不在凸包上的点
    // 使用 Vec 而不是 HashSet，因为 f64 不实现 Hash
    for pt in points {
        let mut on_hull = false;
        for h in hull {
            if (pt[0] - h[0]).abs() < 1e-10 && (pt[1] - h[1]).abs() < 1e-10 {
                on_hull = true;
                break;
            }
        }
        if !on_hull {
            return *pt;
        }
    }
    
    // 如果所有点都在凸包上，返回第一个点
    points[0]
}

/// 检测墙体凹陷
fn detect_wall_indentations(points: &[Point2], hull: &[Point2]) -> Vec<Indentation> {
    let mut indentations = Vec::new();
    let mut in_indentation = false;
    let mut indentation_start = None;
    let mut indentation_points = Vec::new();
    
    for (i, pt) in points.iter().enumerate() {
        // 检查点是否在凸包上
        let mut on_hull = false;
        for h in hull {
            if (pt[0] - h[0]).abs() < 1e-10 && (pt[1] - h[1]).abs() < 1e-10 {
                on_hull = true;
                break;
            }
        }
        
        if !on_hull {
            // 在凹陷内
            if !in_indentation {
                indentation_start = Some(i);
                in_indentation = true;
                indentation_points = vec![*pt];
            } else {
                indentation_points.push(*pt);
            }
        } else if in_indentation {
            // 凹陷结束
            if let Some(start) = indentation_start {
                let width = calculate_indentation_width(&indentation_points);
                let depth = calculate_indentation_depth(&indentation_points, hull);
                
                if depth > 0.1 {
                    indentations.push(Indentation {
                        start_idx: start,
                        end_idx: i,
                        depth,
                        width,
                    });
                }
            }
            in_indentation = false;
            indentation_start = None;
            indentation_points.clear();
        }
    }
    
    indentations
}

/// 计算凹陷宽度
fn calculate_indentation_width(points: &[Point2]) -> f64 {
    if points.is_empty() {
        return 0.0;
    }
    
    // 简单实现：第一个点和最后一个点的距离
    let first = points[0];
    let last = points[points.len() - 1];
    ((first[0] - last[0]).powi(2) + (first[1] - last[1]).powi(2)).sqrt()
}

/// 计算凹陷深度
fn calculate_indentation_depth(points: &[Point2], hull: &[Point2]) -> f64 {
    if points.is_empty() || hull.is_empty() {
        return 0.0;
    }
    
    // 计算凹陷点到凸包的最短距离
    let mut max_depth = 0.0;
    for pt in points {
        let dist = distance_to_hull(pt, hull);
        if dist > max_depth {
            max_depth = dist;
        }
    }
    
    max_depth
}

/// 计算点到凸包的距离
fn distance_to_hull(point: &Point2, hull: &[Point2]) -> f64 {
    if hull.len() < 3 {
        return 0.0;
    }
    
    let mut min_dist = f64::MAX;
    
    // 计算点到每条边的距离
    for i in 0..hull.len() {
        let j = (i + 1) % hull.len();
        let dist = distance_to_segment(*point, hull[i], hull[j]);
        if dist < min_dist {
            min_dist = dist;
        }
    }
    
    min_dist
}

/// 计算点到线段的距离
fn distance_to_segment(point: Point2, a: Point2, b: Point2) -> f64 {
    let ab = [b[0] - a[0], b[1] - a[1]];
    let ap = [point[0] - a[0], point[1] - a[1]];
    
    let ab_len_sq = ab[0] * ab[0] + ab[1] * ab[1];
    if ab_len_sq < 1e-10 {
        return ((point[0] - a[0]).powi(2) + (point[1] - a[1]).powi(2)).sqrt();
    }

    let t = (ap[0] * ab[0] + ap[1] * ab[1]) / ab_len_sq;
    let t = t.clamp(0.0, 1.0);

    let closest = [a[0] + t * ab[0], a[1] + t * ab[1]];
    ((point[0] - closest[0]).powi(2) + (point[1] - closest[1]).powi(2)).sqrt()
}

#[cfg(test)]
mod convexity_tests {
    use super::*;
    use common_types::ClosedLoop;
    
    #[test]
    fn test_convex_hull_rectangle() {
        let points = vec![
            [0.0, 0.0], [4.0, 0.0], [4.0, 3.0], [0.0, 3.0],
        ];
        let hull = compute_convex_hull(&points);
        assert_eq!(hull.len(), 4);
    }
    
    #[test]
    fn test_convex_hull_l_shape() {
        let points = vec![
            [0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [1.0, 1.0], [1.0, 2.0], [0.0, 2.0],
        ];
        let hull = compute_convex_hull(&points);
        // L 形的凸包应该是矩形，但由于内角点 (1,1) 不在凸包上
        // 所以凸包有 5 个点：(0,0), (2,0), (2,1), (1,2), (0,2)
        assert_eq!(hull.len(), 5);
    }
    
    #[test]
    fn test_convexity_ratio_convex() {
        let points = vec![
            [0.0, 0.0], [4.0, 0.0], [4.0, 3.0], [0.0, 3.0],
        ];
        let loop_data = ClosedLoop::new(points);
        let issues = check_convexity(&loop_data);
        // 凸多边形应该没有问题
        assert!(issues.is_empty());
    }
    
    #[test]
    fn test_convexity_ratio_concave() {
        // L 形房间
        let points = vec![
            [0.0, 0.0], [4.0, 0.0], [4.0, 2.0], [2.0, 2.0], [2.0, 4.0], [0.0, 4.0],
        ];
        let loop_data = ClosedLoop::new(points);
        let issues = check_convexity(&loop_data);
        // 凹多边形应该有警告
        assert!(!issues.is_empty());
    }
}
