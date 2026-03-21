//! 矢量化质量评估
//!
//! 提供完整性、准确性、连续性等质量指标

use common_types::{Point2, Polyline};

/// 质量评估报告
#[derive(Debug, Clone)]
pub struct VectorizeQualityReport {
    /// 完整性得分（0-1）：提取的线段数量 / 预期线段数量
    pub completeness: f64,
    /// 准确性得分（0-1）：正确提取的线段 / 总线段数量
    pub accuracy: f64,
    /// 连续性得分（0-1）：无断点的线段比例
    pub continuity: f64,
    /// 总体得分（0-100）
    pub overall_score: f64,
    /// 问题列表
    pub issues: Vec<QualityIssue>,
}

/// 质量问题
#[derive(Debug, Clone)]
pub struct QualityIssue {
    /// 问题类型
    pub issue_type: QualityIssueType,
    /// 严重程度
    pub severity: Severity,
    /// 问题描述
    pub description: String,
    /// 问题位置（如果有）
    pub location: Option<Point2>,
}

/// 问题类型
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QualityIssueType {
    /// 缺失线段
    MissingLines,
    /// 多余噪声
    ExtraNoise,
    /// 断点未连接
    BrokenConnections,
    /// 圆弧拟合错误
    IncorrectArcs,
    /// 比例不匹配
    ScaleMismatch,
    /// 线型识别错误
    LineTypeError,
}

/// 严重程度
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity {
    /// 低
    Low,
    /// 中
    Medium,
    /// 高
    High,
    /// 严重
    Critical,
}

impl VectorizeQualityReport {
    /// 创建空报告
    pub fn empty() -> Self {
        Self {
            completeness: 0.0,
            accuracy: 0.0,
            continuity: 0.0,
            overall_score: 0.0,
            issues: Vec::new(),
        }
    }

    /// 计算总体得分
    pub fn calculate_overall_score(&mut self) {
        self.overall_score = (self.completeness * 0.4 + self.accuracy * 0.4 + self.continuity * 0.2) * 100.0;
    }
}

impl Default for VectorizeQualityReport {
    fn default() -> Self {
        Self::empty()
    }
}

/// 评估矢量化质量
///
/// # 参数
/// - `original`: 原始图像（用于估计预期线段）
/// - `vectorized`: 矢量化结果
///
/// # 返回
/// 质量评估报告
pub fn evaluate_quality(
    _original: &common_types::PdfRasterImage,
    vectorized: &[Polyline],
) -> VectorizeQualityReport {
    // 1. 计算完整性（基于总长度估计）
    let actual_length: f64 = vectorized.iter().map(polyline_length).sum();
    let expected_length = estimate_total_length(vectorized);
    let completeness = if expected_length > 0.0 {
        (actual_length / expected_length).min(1.0)
    } else {
        0.0
    };

    // 2. 计算连续性（检测缺口）
    let gaps = count_gaps(vectorized);
    let total_endpoints = vectorized.len() * 2;
    let continuity = if total_endpoints > 0 {
        1.0 - (gaps as f64 / total_endpoints as f64)
    } else {
        1.0
    };

    // 3. 计算准确性（启发式方法）
    let accuracy = heuristic_accuracy(vectorized);

    // 4. 创建报告
    let mut report = VectorizeQualityReport {
        completeness,
        accuracy,
        continuity,
        overall_score: 0.0,
        issues: Vec::new(),
    };

    // 5. 计算总体得分
    report.calculate_overall_score();

    // 6. 生成问题列表
    generate_issues(&mut report, vectorized);

    report
}

/// 估计总长度（基于矢量化结果）
fn estimate_total_length(_vectorized: &[Polyline]) -> f64 {
    // 简化实现：使用实际长度作为估计
    // TODO: 可以基于图像边缘密度进行更准确的估计
    _vectorized.iter().map(polyline_length).sum()
}

/// 计算多段线长度
fn polyline_length(polyline: &Polyline) -> f64 {
    if polyline.len() < 2 {
        return 0.0;
    }

    polyline
        .windows(2)
        .map(|pts| {
            let dx = pts[1][0] - pts[0][0];
            let dy = pts[1][1] - pts[0][1];
            (dx * dx + dy * dy).sqrt()
        })
        .sum()
}

/// 统计缺口数量
fn count_gaps(polylines: &[Polyline]) -> usize {
    if polylines.is_empty() {
        return 0;
    }

    // 简单实现：统计多段线数量 - 1（假设应该连接成一条）
    // TODO: 更复杂的分析应该考虑端点距离
    polylines.len().saturating_sub(1)
}

/// 启发式准确性评估
fn heuristic_accuracy(vectorized: &[Polyline]) -> f64 {
    if vectorized.is_empty() {
        return 0.0;
    }

    let mut score: f64 = 1.0;

    // 1. 检查线段数量
    let segment_count = vectorized.len();
    if segment_count < 5 {
        score -= 0.2; // 线段太少
    }

    // 2. 检查线段长度分布
    let lengths: Vec<f64> = vectorized.iter().map(polyline_length).collect();
    let avg_len = lengths.iter().sum::<f64>() / lengths.len() as f64;
    let variance: f64 = lengths
        .iter()
        .map(|&l| (l - avg_len).powi(2))
        .sum::<f64>() / lengths.len() as f64;
    let std_dev = variance.sqrt();

    // 如果长度变化太大，可能是噪声
    if std_dev > avg_len * 2.0 {
        score -= 0.1;
    }

    // 3. 检查是否有过短的线段（可能是噪声）
    let short_segments = lengths.iter().filter(|&&l| l < 5.0).count();
    if short_segments > segment_count / 2 {
        score -= 0.2;
    }

    score.max(0.0)
}

/// 生成问题列表
fn generate_issues(report: &mut VectorizeQualityReport, vectorized: &[Polyline]) {
    // 1. 检查完整性
    if report.completeness < 0.5 {
        report.issues.push(QualityIssue {
            issue_type: QualityIssueType::MissingLines,
            severity: Severity::High,
            description: format!("完整性过低 ({:.1}%)，可能缺失大量线段", report.completeness * 100.0),
            location: None,
        });
    }

    // 2. 检查连续性
    if report.continuity < 0.8 {
        report.issues.push(QualityIssue {
            issue_type: QualityIssueType::BrokenConnections,
            severity: Severity::Medium,
            description: format!("连续性较低 ({:.1}%)，存在多处断点", report.continuity * 100.0),
            location: None,
        });
    }

    // 3. 检查噪声
    let short_count = vectorized
        .iter()
        .filter(|pl| polyline_length(pl) < 5.0)
        .count();
    if short_count > vectorized.len() / 3 {
        report.issues.push(QualityIssue {
            issue_type: QualityIssueType::ExtraNoise,
            severity: Severity::Medium,
            description: format!("检测到 {} 条过短线段，可能存在噪声", short_count),
            location: None,
        });
    }

    // 4. 检查空结果
    if vectorized.is_empty() {
        report.issues.push(QualityIssue {
            issue_type: QualityIssueType::MissingLines,
            severity: Severity::Critical,
            description: "未提取到任何线段".to_string(),
            location: None,
        });
    }
}

/// 质量评估等级
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QualityGrade {
    /// 优秀 (>= 90)
    Excellent,
    /// 良好 (>= 75)
    Good,
    /// 合格 (>= 60)
    Pass,
    /// 不合格 (< 60)
    Fail,
}

impl QualityGrade {
    pub fn from_score(score: f64) -> Self {
        if score >= 90.0 {
            QualityGrade::Excellent
        } else if score >= 75.0 {
            QualityGrade::Good
        } else if score >= 60.0 {
            QualityGrade::Pass
        } else {
            QualityGrade::Fail
        }
    }
}

impl std::fmt::Display for QualityGrade {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            QualityGrade::Excellent => write!(f, "优秀"),
            QualityGrade::Good => write!(f, "良好"),
            QualityGrade::Pass => write!(f, "合格"),
            QualityGrade::Fail => write!(f, "不合格"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use common_types::PdfRasterImage;

    fn create_test_raster_image() -> PdfRasterImage {
        PdfRasterImage::new(
            "test".to_string(),
            100,
            100,
            vec![0u8; 100 * 100],
            Some((72.0, 72.0)),
            [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        )
    }

    #[test]
    fn test_evaluate_quality_good_result() {
        let image = create_test_raster_image();
        let vectorized = vec![
            vec![[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]],
            vec![[0.0, 10.0], [10.0, 10.0], [20.0, 10.0]],
            vec![[0.0, 20.0], [10.0, 20.0], [20.0, 20.0]],
        ];

        let report = evaluate_quality(&image, &vectorized);

        assert!(report.overall_score > 0.0);
        assert!(report.overall_score <= 100.0);
    }

    #[test]
    fn test_evaluate_quality_empty_result() {
        let image = create_test_raster_image();
        let vectorized: Vec<Polyline> = vec![];

        let report = evaluate_quality(&image, &vectorized);

        assert_eq!(report.completeness, 0.0);
        assert_eq!(report.accuracy, 0.0);
        assert!(!report.issues.is_empty());
    }

    #[test]
    fn test_quality_grade() {
        assert_eq!(QualityGrade::from_score(95.0), QualityGrade::Excellent);
        assert_eq!(QualityGrade::from_score(80.0), QualityGrade::Good);
        assert_eq!(QualityGrade::from_score(65.0), QualityGrade::Pass);
        assert_eq!(QualityGrade::from_score(50.0), QualityGrade::Fail);
    }

    #[test]
    fn test_polyline_length() {
        let polyline = vec![[0.0, 0.0], [3.0, 0.0], [3.0, 4.0]];
        let length = polyline_length(&polyline);
        assert!((length - 7.0).abs() < 1e-10); // 3 + 4 = 7
    }
}
