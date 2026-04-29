//! VLM (Vision-Language Model) 尺寸标注解析模块
//!
//! 使用多模态大模型解析工程图纸中的复杂尺寸标注：
//! - 尺寸数值、公差、形位公差识别
//! - 基准符号识别
//! - 表面粗糙度识别
//! - 装配关系标注识别

use image::GrayImage;
use std::fmt;

use super::ocr::{TextRecognition, TextType};

/// 公差类型
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToleranceType {
    /// 对称公差 ±X
    Symmetric,
    /// 上下偏差 +X/-Y
    Limit,
    /// 最大实体要求 M
    MaximumMaterial,
    /// 最小实体要求 L
    LeastMaterial,
    /// 包容要求 E
    Envelope,
    /// 独立原则 R
    Regardless,
}

impl fmt::Display for ToleranceType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ToleranceType::Symmetric => write!(f, "对称公差"),
            ToleranceType::Limit => write!(f, "极限偏差"),
            ToleranceType::MaximumMaterial => write!(f, "最大实体"),
            ToleranceType::LeastMaterial => write!(f, "最小实体"),
            ToleranceType::Envelope => write!(f, "包容要求"),
            ToleranceType::Regardless => write!(f, "独立原则"),
        }
    }
}

/// 形位公差类型
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GeometricToleranceType {
    /// 直线度
    Straightness,
    /// 平面度
    Flatness,
    /// 圆度
    Circularity,
    /// 圆柱度
    Cylindricity,
    /// 线轮廓度
    ProfileLine,
    /// 面轮廓度
    ProfileSurface,
    /// 平行度
    Parallelism,
    /// 垂直度
    Perpendicularity,
    /// 倾斜度
    Angularity,
    /// 位置度
    Position,
    /// 同轴度
    Concentricity,
    /// 对称度
    Symmetry,
    /// 圆跳动
    CircularRunout,
    /// 全跳动
    TotalRunout,
}

impl fmt::Display for GeometricToleranceType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GeometricToleranceType::Straightness => write!(f, "直线度"),
            GeometricToleranceType::Flatness => write!(f, "平面度"),
            GeometricToleranceType::Circularity => write!(f, "圆度"),
            GeometricToleranceType::Cylindricity => write!(f, "圆柱度"),
            GeometricToleranceType::ProfileLine => write!(f, "线轮廓度"),
            GeometricToleranceType::ProfileSurface => write!(f, "面轮廓度"),
            GeometricToleranceType::Parallelism => write!(f, "平行度"),
            GeometricToleranceType::Perpendicularity => write!(f, "垂直度"),
            GeometricToleranceType::Angularity => write!(f, "倾斜度"),
            GeometricToleranceType::Position => write!(f, "位置度"),
            GeometricToleranceType::Concentricity => write!(f, "同轴度"),
            GeometricToleranceType::Symmetry => write!(f, "对称度"),
            GeometricToleranceType::CircularRunout => write!(f, "圆跳动"),
            GeometricToleranceType::TotalRunout => write!(f, "全跳动"),
        }
    }
}

/// 解析后的尺寸标注
#[derive(Debug, Clone)]
pub struct DimensionAnnotation {
    /// 原始识别文本
    pub raw_text: String,
    /// 尺寸标称值
    pub nominal_value: Option<f64>,
    /// 公差类型
    pub tolerance_type: Option<ToleranceType>,
    /// 上偏差
    pub upper_deviation: Option<f64>,
    /// 下偏差
    pub lower_deviation: Option<f64>,
    /// 形位公差类型
    pub geometric_type: Option<GeometricToleranceType>,
    /// 基准符号
    pub datums: Vec<String>,
    /// 表面粗糙度值
    pub roughness: Option<f64>,
    /// 解析置信度
    pub confidence: f64,
}

impl Default for DimensionAnnotation {
    fn default() -> Self {
        Self {
            raw_text: String::new(),
            nominal_value: None,
            tolerance_type: None,
            upper_deviation: None,
            lower_deviation: None,
            geometric_type: None,
            datums: Vec::new(),
            roughness: None,
            confidence: 0.0,
        }
    }
}

/// VLM 后端 trait（可插拔接口）
pub trait VlmBackend: Send + Sync {
    /// 从灰度图像解析尺寸标注
    fn parse_dimension(&self, image: &GrayImage) -> DimensionAnnotation;

    /// 批量解析多个文本区域
    fn parse_batch(&self, images: &[GrayImage]) -> Vec<DimensionAnnotation> {
        images.iter().map(|img| self.parse_dimension(img)).collect()
    }

    /// 后端名称
    fn name(&self) -> &str;
}

/// 纯 Rust 启发式 VLM 后端（无需外部依赖）
///
/// 适用于：
/// - CI/CD 环境
/// - 无法安装 ML 框架的系统
/// - 标准格式尺寸标注解析
///
/// 基于正则表达式和规则引擎解析标准标注格式
#[derive(Default)]
pub struct HeuristicVlmBackend;

impl HeuristicVlmBackend {
    pub fn new() -> Self {
        Self
    }

    /// 从文本解析尺寸标注
    pub fn parse_text(&self, text: &str) -> DimensionAnnotation {
        let trimmed = text.trim();
        let mut result = DimensionAnnotation {
            raw_text: trimmed.to_string(),
            ..Default::default()
        };

        let mut confidence = 0.3;

        // 解析标称值（如 100, 50.5, φ20）
        if let Ok(val) = trimmed.replace('φ', "").parse::<f64>() {
            result.nominal_value = Some(val);
            confidence += 0.3;
        } else {
            // 尝试提取数字
            let num_regex = regex::Regex::new(r"(\d+\.?\d*)").unwrap();
            if let Some(cap) = num_regex.captures(trimmed) {
                if let Ok(val) = cap[1].parse::<f64>() {
                    result.nominal_value = Some(val);
                    confidence += 0.2;
                }
            }
        }

        // 检测对称公差 ±
        if trimmed.contains('±') {
            result.tolerance_type = Some(ToleranceType::Symmetric);
            confidence += 0.1;
            // 提取公差值
            let parts: Vec<&str> = trimmed.split('±').collect();
            if parts.len() > 1 {
                if let Ok(tol) = parts[1].parse::<f64>() {
                    result.upper_deviation = Some(tol);
                    result.lower_deviation = Some(-tol);
                }
            }
        }

        // 检测上下偏差 +X/-Y 格式
        if trimmed.contains('+') && trimmed.contains('-') {
            result.tolerance_type = Some(ToleranceType::Limit);
            confidence += 0.1;
        }

        // 检测螺纹标注 M20, M6 等
        if trimmed.starts_with('M') || trimmed.starts_with('m') {
            let num_part = trimmed.trim_start_matches(|c: char| !c.is_ascii_digit());
            if let Ok(val) = num_part.parse::<f64>() {
                result.nominal_value = Some(val);
                confidence += 0.1;
            }
        }

        // 检测基准符号 A, B, C 等
        let datum_regex = regex::Regex::new(r"\b([A-Z])\b").unwrap();
        for cap in datum_regex.captures_iter(trimmed) {
            result.datums.push(cap[1].to_string());
        }
        if !result.datums.is_empty() {
            confidence += 0.1;
        }

        // 检测粗糙度符号 Ra
        if trimmed.contains("Ra") || trimmed.contains("ra") {
            let ra_regex = regex::Regex::new(r"Ra\.?\s*(\d+\.?\d*)").unwrap();
            if let Some(cap) = ra_regex.captures(trimmed) {
                if let Ok(val) = cap[1].parse::<f64>() {
                    result.roughness = Some(val);
                    confidence += 0.2;
                }
            }
        }

        // 形位公差符号检测（基于文本模式）
        let geometric_patterns = [
            ("同轴度", GeometricToleranceType::Concentricity),
            ("对称度", GeometricToleranceType::Symmetry),
            ("位置度", GeometricToleranceType::Position),
            ("平行度", GeometricToleranceType::Parallelism),
            ("垂直度", GeometricToleranceType::Perpendicularity),
            ("倾斜度", GeometricToleranceType::Angularity),
            ("圆跳动", GeometricToleranceType::CircularRunout),
            ("全跳动", GeometricToleranceType::TotalRunout),
            ("直线度", GeometricToleranceType::Straightness),
            ("平面度", GeometricToleranceType::Flatness),
            ("圆度", GeometricToleranceType::Circularity),
            ("圆柱度", GeometricToleranceType::Cylindricity),
        ];

        for (pattern, geo_type) in geometric_patterns.iter() {
            if trimmed.contains(pattern) {
                result.geometric_type = Some(*geo_type);
                confidence += 0.1;
                break;
            }
        }

        result.confidence = f64::min(confidence, 1.0);
        result
    }
}

impl VlmBackend for HeuristicVlmBackend {
    fn parse_dimension(&self, _image: &GrayImage) -> DimensionAnnotation {
        // 启发式实现：返回默认空解析
        // 实际使用时需要先 OCR 识别文本，再调用 parse_text
        DimensionAnnotation::default()
    }

    fn name(&self) -> &str {
        "HeuristicVLM (Pure Rust Rule-based)"
    }
}

/// 尺寸标注分析器
pub struct DimensionAnalyzer {
    _vlm_backend: Box<dyn VlmBackend>,
}

impl Default for DimensionAnalyzer {
    fn default() -> Self {
        Self {
            _vlm_backend: Box::new(HeuristicVlmBackend::new()),
        }
    }
}

impl DimensionAnalyzer {
    pub fn new(backend: Box<dyn VlmBackend>) -> Self {
        Self {
            _vlm_backend: backend,
        }
    }

    /// 从 OCR 识别结果批量解析尺寸标注
    pub fn analyze_recognitions(
        &self,
        recognitions: &[TextRecognition],
    ) -> Vec<DimensionAnnotation> {
        let heuristic = HeuristicVlmBackend::new();
        recognitions
            .iter()
            .map(|rec| {
                let mut dim = heuristic.parse_text(&rec.text);
                dim.confidence *= rec.confidence;
                dim
            })
            .collect()
    }

    /// 分类文本类型（扩展 OCR 的分类）
    pub fn classify_dimension_type(&self, text: &str) -> TextType {
        let trimmed = text.trim();

        if trimmed.contains('±')
            || (trimmed.starts_with('M')
                && trimmed[1..]
                    .chars()
                    .next()
                    .is_some_and(|c| c.is_ascii_digit()))
        {
            return TextType::Tolerance;
        }

        if trimmed.contains("Ra") || trimmed.contains("粗糙度") || trimmed.contains("不平度")
        {
            return TextType::TechnicalNote;
        }

        TextType::Other
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_dimension() {
        let vlm = HeuristicVlmBackend::new();
        let dim = vlm.parse_text("100");
        assert_eq!(dim.nominal_value, Some(100.0));
        assert!(dim.confidence > 0.5);
    }

    #[test]
    fn test_parse_symmetric_tolerance() {
        let vlm = HeuristicVlmBackend::new();
        let dim = vlm.parse_text("100±0.5");
        assert_eq!(dim.nominal_value, Some(100.0));
        assert_eq!(dim.tolerance_type, Some(ToleranceType::Symmetric));
        assert_eq!(dim.upper_deviation, Some(0.5));
        assert_eq!(dim.lower_deviation, Some(-0.5));
    }

    #[test]
    fn test_parse_thread_dimension() {
        let vlm = HeuristicVlmBackend::new();
        let dim = vlm.parse_text("M20");
        assert_eq!(dim.nominal_value, Some(20.0));
    }

    #[test]
    fn test_parse_roughness() {
        let vlm = HeuristicVlmBackend::new();
        let dim = vlm.parse_text("Ra3.2");
        assert_eq!(dim.roughness, Some(3.2));
    }

    #[test]
    fn test_parse_with_datum() {
        let vlm = HeuristicVlmBackend::new();
        let dim = vlm.parse_text("⊥0.05 A");
        assert!(dim.datums.contains(&"A".to_string()));
    }

    #[test]
    fn test_analyzer_classification() {
        let analyzer = DimensionAnalyzer::default();
        assert_eq!(
            analyzer.classify_dimension_type("100±0.5"),
            TextType::Tolerance
        );
        assert_eq!(
            analyzer.classify_dimension_type("M20-6H"),
            TextType::Tolerance
        );
        assert_eq!(
            analyzer.classify_dimension_type("Ra1.6"),
            TextType::TechnicalNote
        );
    }

    #[test]
    fn test_vlm_backend_name() {
        let vlm = HeuristicVlmBackend::new();
        assert_eq!(vlm.name(), "HeuristicVLM (Pure Rust Rule-based)");
    }

    #[test]
    fn test_default_dimension_annotation() {
        let dim = DimensionAnnotation::default();
        assert!(dim.raw_text.is_empty());
        assert!(dim.nominal_value.is_none());
        assert_eq!(dim.confidence, 0.0);
    }

    #[test]
    fn test_tolerance_type_display() {
        assert_eq!(ToleranceType::Symmetric.to_string(), "对称公差");
        assert_eq!(GeometricToleranceType::Parallelism.to_string(), "平行度");
    }
}
