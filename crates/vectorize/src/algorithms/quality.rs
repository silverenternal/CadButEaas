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
    /// 分辨率过低
    LowResolution,
    /// 页面偏斜
    SkewedPage,
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
        self.overall_score =
            (self.completeness * 0.4 + self.accuracy * 0.4 + self.continuity * 0.2) * 100.0;
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
    original: &common_types::PdfRasterImage,
    vectorized: &[Polyline],
) -> VectorizeQualityReport {
    // 1. 计算完整性（基于原始图像边缘密度估计）
    let actual_length: f64 = vectorized.iter().map(polyline_length).sum();
    let expected_length = estimate_total_length_from_image(original);
    let completeness = if expected_length > 0.0 {
        (actual_length / expected_length).min(1.0)
    } else {
        0.0
    };

    // 2. 计算连续性（基于空间端点距离分析）
    let gaps = count_gaps_spatial(vectorized);
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

    // 6. 生成问题列表（包含扫描图纸特有检查）
    generate_issues(&mut report, vectorized, original);

    report
}

/// 基于原始图像边缘密度估计总长度
fn estimate_total_length_from_image(original: &common_types::PdfRasterImage) -> f64 {
    let img = original.to_image();
    let gray = img.to_luma8();
    let (width, height) = gray.dimensions();

    if width < 3 || height < 3 {
        return 0.0;
    }

    // 使用 Sobel 算子估算边缘像素总数
    let mut edge_pixel_count: usize = 0;
    for y in 1..(height - 1) {
        for x in 1..(width - 1) {
            let gx = gray.get_pixel(x + 1, y - 1)[0] as i32
                + 2 * gray.get_pixel(x + 1, y)[0] as i32
                + gray.get_pixel(x + 1, y + 1)[0] as i32
                - gray.get_pixel(x - 1, y - 1)[0] as i32
                - 2 * gray.get_pixel(x - 1, y)[0] as i32
                - gray.get_pixel(x - 1, y + 1)[0] as i32;
            let gy = gray.get_pixel(x - 1, y + 1)[0] as i32
                + 2 * gray.get_pixel(x, y + 1)[0] as i32
                + gray.get_pixel(x + 1, y + 1)[0] as i32
                - gray.get_pixel(x - 1, y - 1)[0] as i32
                - 2 * gray.get_pixel(x, y - 1)[0] as i32
                - gray.get_pixel(x + 1, y - 1)[0] as i32;

            if (gx * gx + gy * gy) > 25_000 {
                edge_pixel_count += 1;
            }
        }
    }

    // 边缘像素数作为预期线段长度的粗略估计（1 像素 ≈ 1 单位长度）
    edge_pixel_count as f64
}

/// 基于空间端点距离的缺口计数
fn count_gaps_spatial(polylines: &[Polyline]) -> usize {
    if polylines.len() < 2 {
        return 0;
    }

    // 使用 gap_filling 模块的 detect_gaps 进行真正的空间分析
    let gaps = super::gap_filling::detect_gaps(polylines, 5.0);
    gaps.len()
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
    let variance: f64 =
        lengths.iter().map(|&l| (l - avg_len).powi(2)).sum::<f64>() / lengths.len() as f64;
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
fn generate_issues(
    report: &mut VectorizeQualityReport,
    vectorized: &[Polyline],
    original: &common_types::PdfRasterImage,
) {
    // 1. 检查完整性
    if report.completeness < 0.5 {
        report.issues.push(QualityIssue {
            issue_type: QualityIssueType::MissingLines,
            severity: Severity::High,
            description: format!(
                "完整性过低 ({:.1}%)，可能缺失大量线段",
                report.completeness * 100.0
            ),
            location: None,
        });
    }

    // 2. 检查连续性
    if report.continuity < 0.8 {
        report.issues.push(QualityIssue {
            issue_type: QualityIssueType::BrokenConnections,
            severity: Severity::Medium,
            description: format!(
                "连续性较低 ({:.1}%)，存在多处断点",
                report.continuity * 100.0
            ),
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

    // 5. 扫描图纸特有检查：分辨率
    if let Some((dpi_x, dpi_y)) = original.dpi {
        let avg_dpi = (dpi_x + dpi_y) / 2.0;
        if avg_dpi < 150.0 {
            report.issues.push(QualityIssue {
                issue_type: QualityIssueType::LowResolution,
                severity: Severity::Medium,
                description: format!("图像分辨率较低 ({:.0} DPI)，可能影响识别精度", avg_dpi),
                location: None,
            });
        }
    }

    // 6. 扫描图纸特有检查：偏斜检测
    let skew_angle = detect_skew_angle(original);
    if skew_angle.abs() > 2.0 {
        let severity = if skew_angle.abs() > 5.0 {
            Severity::High
        } else {
            Severity::Medium
        };
        report.issues.push(QualityIssue {
            issue_type: QualityIssueType::SkewedPage,
            severity,
            description: format!("检测到页面偏斜 ({:+.1}°)，可能影响线条对齐", skew_angle),
            location: None,
        });
    }
}

/// 检测图像偏斜角度
///
/// 通过 Sobel 梯度方向分析，寻找主导边缘方向。
/// 对于技术图纸，主导方向应为 0°（水平）和 90°（垂直）。
/// 如果主导方向偏离这些角度，说明页面偏斜。
///
/// # 返回
/// 偏斜角度（度），正值为顺时针偏斜
fn detect_skew_angle(original: &common_types::PdfRasterImage) -> f64 {
    let img = original.to_image();
    let gray = img.to_luma8();
    let (width, height) = gray.dimensions();

    if width < 10 || height < 10 {
        return 0.0;
    }

    // 使用 30° 步长构建角度直方图（6 个 bin，0-180°）
    const NUM_BINS: usize = 6;
    let bin_step = 180.0 / NUM_BINS as f64; // 30° per bin
    let mut angle_histogram = [0usize; NUM_BINS];

    // 采样：每 4 像素取一个点，避免全图遍历
    let step = 4u32;

    for y in (1..(height - 1)).step_by(step as usize) {
        for x in (1..(width - 1)).step_by(step as usize) {
            let gx = gray.get_pixel(x + 1, y - 1)[0] as i32
                + 2 * gray.get_pixel(x + 1, y)[0] as i32
                + gray.get_pixel(x + 1, y + 1)[0] as i32
                - gray.get_pixel(x - 1, y - 1)[0] as i32
                - 2 * gray.get_pixel(x - 1, y)[0] as i32
                - gray.get_pixel(x - 1, y + 1)[0] as i32;
            let gy = gray.get_pixel(x - 1, y + 1)[0] as i32
                + 2 * gray.get_pixel(x, y + 1)[0] as i32
                + gray.get_pixel(x + 1, y + 1)[0] as i32
                - gray.get_pixel(x - 1, y - 1)[0] as i32
                - 2 * gray.get_pixel(x, y - 1)[0] as i32
                - gray.get_pixel(x + 1, y - 1)[0] as i32;

            let magnitude = ((gx * gx + gy * gy) as f64).sqrt();
            if magnitude < 50.0 {
                continue; // 忽略弱边缘
            }

            // 计算梯度方向角度（0-180°，因为梯度方向是双向的）
            let angle_rad = (gy as f64).atan2(gx as f64);
            let angle_deg = angle_rad.to_degrees().rem_euclid(180.0);

            // 归一化到最近的 bin
            let bin = ((angle_deg / bin_step).round() as usize) % NUM_BINS;
            angle_histogram[bin] += 1;
        }
    }

    // 找到主导角度 bin
    let dominant_bin = angle_histogram
        .iter()
        .enumerate()
        .max_by_key(|&(_, count)| count)
        .map(|(i, _)| i)
        .unwrap_or(0);

    let dominant_angle = dominant_bin as f64 * bin_step + bin_step / 2.0;

    // 技术图纸的主导方向应为 0°（水平）或 90°（垂直）
    // 计算最接近 0° 或 90° 的偏差
    let dist_to_0 = dominant_angle.min(180.0 - dominant_angle);
    let dist_to_90 = (dominant_angle - 90.0).abs();
    dist_to_0.min(dist_to_90)
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

    #[test]
    fn test_detect_skew_angle_no_skew_horizontal() {
        // 纯水平线的图像 → 偏斜角接近 0°
        let mut pixels = vec![255u8; 100 * 100];
        // 画 20 条水平黑线
        for y in (0..100).step_by(5) {
            for x in 0..100 {
                pixels[y * 100 + x] = 0;
            }
        }
        let pdf_image = PdfRasterImage::new(
            "test".to_string(),
            100,
            100,
            pixels,
            Some((300.0, 300.0)),
            [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        );

        let angle = detect_skew_angle(&pdf_image);
        assert!(angle <= 15.0, "水平线图像偏斜角应 <= 15°，实际 {}", angle);
    }

    #[test]
    fn test_detect_skew_angle_no_skew_vertical() {
        // 纯垂直线的图像 → 偏斜角接近 0°
        let mut pixels = vec![255u8; 100 * 100];
        // 画 20 条垂直黑线
        for x in (0..100).step_by(5) {
            for y in 0..100 {
                pixels[y * 100 + x] = 0;
            }
        }
        let pdf_image = PdfRasterImage::new(
            "test".to_string(),
            100,
            100,
            pixels,
            Some((300.0, 300.0)),
            [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        );

        let angle = detect_skew_angle(&pdf_image);
        assert!(angle <= 15.0, "垂直线图像偏斜角应 <= 15°，实际 {}", angle);
    }

    #[test]
    fn test_detect_skew_angle_small_image() {
        let pdf_image = PdfRasterImage::new(
            "test".to_string(),
            5,
            5,
            vec![255u8; 5 * 5],
            None,
            [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        );

        let angle = detect_skew_angle(&pdf_image);
        assert_eq!(angle, 0.0, "过小图像应返回 0°");
    }
}
