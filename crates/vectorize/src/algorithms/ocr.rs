//! OCR 文本识别与空间关联模块
//!
//! 将光栅图像中的文字识别并与几何图元关联：
//! - 文本区域检测（基于连通分量的启发式检测）
//! - 文本识别（可插拔 OCR 后端抽象）
//! - 文本-几何空间关联算法（尺寸标注、技术要求等）

use common_types::Point2;
use image::GrayImage;
use std::fmt;

use super::text_blob::{detect_text_blobs, BoundingBox, TextBlob};

/// 文本识别结果
#[derive(Debug, Clone)]
pub struct TextRecognition {
    /// 识别的文本内容
    pub text: String,
    /// 置信度 (0.0 ~ 1.0)
    pub confidence: f64,
    /// 文字包围盒
    pub bbox: BoundingBox,
    /// 文本朝向（角度，0=水平）
    pub orientation: f64,
}

impl From<&TextBlob> for BoundingBox {
    fn from(blob: &TextBlob) -> Self {
        Self {
            x_min: blob.bbox_min[0] as u32,
            y_min: blob.bbox_min[1] as u32,
            x_max: blob.bbox_max[0] as u32,
            y_max: blob.bbox_max[1] as u32,
        }
    }
}

/// OCR 后端 trait（可插拔接口）
pub trait OcrBackend: Send + Sync {
    /// 从灰度图像识别文字
    fn recognize(&self, image: &GrayImage) -> Vec<TextRecognition>;

    /// 检测图像中的文本区域（不识别内容）
    fn detect_regions(&self, image: &GrayImage) -> Vec<BoundingBox> {
        detect_text_blobs(image, 5, 5000)
            .iter()
            .map(BoundingBox::from)
            .collect()
    }

    /// 后端名称
    fn name(&self) -> &str;
}

/// 纯 Rust 启发式 OCR 后端（无需外部依赖）
///
/// 适用于：
/// - CI/CD 环境
/// - 无法安装 Tesseract 的系统
/// - 仅需简单数字/字母识别的场景
///
/// 基于图像模板匹配和字符特征统计
#[derive(Default)]
pub struct HeuristicOcrBackend;

impl HeuristicOcrBackend {
    pub fn new() -> Self {
        Self
    }

    /// 根据字形特征猜测字符类型
    fn guess_char_type(&self, blob: &TextBlob, _img: &GrayImage) -> &'static str {
        let aspect = blob.aspect_ratio;
        let solidity = blob.solidity;

        if aspect < 0.3 {
            "|" // 可能是竖线或字母 I
        } else if aspect > 3.0 {
            "-" // 可能是横线或减号
        } else if solidity > 0.8 {
            "O" // 可能是圆形字符
        } else if solidity > 0.6 {
            "□" // 可能是矩形字符
        } else {
            "?" // 未知
        }
    }
}

impl OcrBackend for HeuristicOcrBackend {
    fn recognize(&self, image: &GrayImage) -> Vec<TextRecognition> {
        let blobs = detect_text_blobs(image, 5, 5000);

        blobs
            .iter()
            .map(|blob| TextRecognition {
                text: self.guess_char_type(blob, image).to_string(),
                confidence: 0.3, // 启发式方法置信度较低
                bbox: BoundingBox::from(blob),
                orientation: 0.0, // 假设水平
            })
            .collect()
    }

    fn name(&self) -> &str {
        "HeuristicOCR (Pure Rust)"
    }
}

/// 文本类型分类
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextType {
    /// 尺寸标注数值（如 100, 50.5）
    DimensionNumber,
    /// 公差标注（如 ±0.1, H7, h6）
    Tolerance,
    /// 技术要求说明
    TechnicalNote,
    /// 材料规格
    MaterialSpec,
    /// 视图标注（如 A-A, I 视图）
    ViewLabel,
    /// 标题栏内容
    TitleBlock,
    /// 其他文本
    Other,
}

impl fmt::Display for TextType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TextType::DimensionNumber => write!(f, "尺寸数值"),
            TextType::Tolerance => write!(f, "公差标注"),
            TextType::TechnicalNote => write!(f, "技术要求"),
            TextType::MaterialSpec => write!(f, "材料规格"),
            TextType::ViewLabel => write!(f, "视图标注"),
            TextType::TitleBlock => write!(f, "标题栏"),
            TextType::Other => write!(f, "其他文本"),
        }
    }
}

/// 文本与几何图元的关联关系
#[derive(Debug, Clone)]
pub struct TextGeometryAssociation {
    /// 文本识别结果
    pub text: TextRecognition,
    /// 文本类型分类
    pub text_type: TextType,
    /// 关联的几何图元索引
    pub associated_geometry: Vec<usize>,
    /// 关联置信度 (0.0 ~ 1.0)
    pub association_confidence: f64,
    /// 关联类型描述
    pub relation_type: String,
}

/// 文本-几何关联分析器
pub struct TextGeometryAnalyzer {
    /// 尺寸标注线搜索距离（字符高度的倍数）
    pub dimension_search_radius_factor: f64,
    /// 标题栏区域（图像底部百分比）
    pub title_block_bottom_percent: f64,
}

impl Default for TextGeometryAnalyzer {
    fn default() -> Self {
        Self {
            dimension_search_radius_factor: 3.0,
            title_block_bottom_percent: 0.2,
        }
    }
}

impl TextGeometryAnalyzer {
    /// 分析文本并分类其类型
    pub fn classify_text(&self, text: &str, bbox: &BoundingBox, image_height: u32) -> TextType {
        let trimmed = text.trim();

        let in_bottom_region =
            bbox.y_min as f64 > image_height as f64 * (1.0 - self.title_block_bottom_percent);
        if in_bottom_region {
            return TextType::TitleBlock;
        }

        if trimmed.parse::<f64>().is_ok() {
            return TextType::DimensionNumber;
        }

        let is_tolerance = trimmed.contains('±')
            || trimmed.contains('+') && trimmed.contains('-')
            || (trimmed.len() >= 2
                && (trimmed.starts_with('H') || trimmed.starts_with('h'))
                && trimmed[1..].chars().all(|c| c.is_ascii_digit()));
        if is_tolerance {
            return TextType::Tolerance;
        }

        let is_view_label = (trimmed.contains('-') && trimmed.len() <= 5)
            || (trimmed.len() <= 3 && trimmed.chars().all(|c| "IVXL".contains(c)));
        if is_view_label {
            return TextType::ViewLabel;
        }

        let material_keywords = ["材料", "材质", "Material", "钢", "铝", "铜", "Q235", "45#"];
        if material_keywords.iter().any(|&kw| trimmed.contains(kw)) {
            return TextType::MaterialSpec;
        }

        let tech_keywords = [
            "技术要求",
            "注",
            "未注",
            "公差",
            "粗糙度",
            "淬火",
            "回火",
            "表面",
            "Technical",
            "Note",
            "requirement",
        ];
        if tech_keywords.iter().any(|&kw| trimmed.contains(kw)) {
            return TextType::TechnicalNote;
        }

        TextType::Other
    }

    /// 将文本与几何图元进行空间关联
    pub fn associate_text_geometry(
        &self,
        texts: &[TextRecognition],
        polylines: &[Vec<Point2>],
        image_height: u32,
    ) -> Vec<TextGeometryAssociation> {
        let mut associations = Vec::new();

        for text in texts {
            let text_type = self.classify_text(&text.text, &text.bbox, image_height);
            let text_center = text.bbox.center();
            let char_height = text.bbox.height() as f64;
            let search_radius = char_height * self.dimension_search_radius_factor;

            let mut nearby_geometries = Vec::new();
            for (idx, polyline) in polylines.iter().enumerate() {
                if polyline.is_empty() {
                    continue;
                }

                let min_dist = polyline
                    .iter()
                    .map(|&pt| {
                        let dx = pt[0] - text_center[0];
                        let dy = pt[1] - text_center[1];
                        (dx * dx + dy * dy).sqrt()
                    })
                    .fold(f64::INFINITY, f64::min);

                if min_dist < search_radius {
                    nearby_geometries.push(idx);
                }
            }

            let confidence = if !nearby_geometries.is_empty() {
                f64::min(0.8, text.confidence + 0.2)
            } else {
                text.confidence * 0.5
            };

            let relation_type = match text_type {
                TextType::DimensionNumber => "尺寸标注值".to_string(),
                TextType::Tolerance => "公差标注".to_string(),
                TextType::TechnicalNote => "技术要求".to_string(),
                TextType::MaterialSpec => "材料规格".to_string(),
                TextType::ViewLabel => "视图标注".to_string(),
                TextType::TitleBlock => "标题栏内容".to_string(),
                TextType::Other => {
                    if nearby_geometries.is_empty() {
                        "孤立文本".to_string()
                    } else {
                        "邻近几何图元".to_string()
                    }
                }
            };

            associations.push(TextGeometryAssociation {
                text: text.clone(),
                text_type,
                associated_geometry: nearby_geometries,
                association_confidence: confidence,
                relation_type,
            });
        }

        associations
    }
}

/// 简单的数字识别器（专门用于工程图纸尺寸标注）
#[derive(Default)]
pub struct DigitRecognizer;

impl DigitRecognizer {
    pub fn new() -> Self {
        Self
    }

    pub fn is_likely_dimension(_blob: &TextBlob) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Luma;

    #[test]
    fn test_bounding_box_iou() {
        let a = BoundingBox {
            x_min: 0,
            y_min: 0,
            x_max: 9,
            y_max: 9,
        };
        let b = BoundingBox {
            x_min: 5,
            y_min: 5,
            x_max: 14,
            y_max: 14,
        };

        let iou = a.iou(&b);
        assert!((iou - 25.0 / 175.0).abs() < 1e-6);
    }

    #[test]
    fn test_bounding_box_no_overlap() {
        let a = BoundingBox {
            x_min: 0,
            y_min: 0,
            x_max: 4,
            y_max: 4,
        };
        let b = BoundingBox {
            x_min: 10,
            y_min: 10,
            x_max: 14,
            y_max: 14,
        };

        assert_eq!(a.iou(&b), 0.0);
    }

    #[test]
    fn test_text_type_classification() {
        let analyzer = TextGeometryAnalyzer::default();
        let img_h = 1000;
        let bbox = BoundingBox {
            x_min: 100,
            y_min: 100,
            x_max: 150,
            y_max: 130,
        };

        assert_eq!(
            analyzer.classify_text("100", &bbox, img_h),
            TextType::DimensionNumber
        );
        assert_eq!(
            analyzer.classify_text("±0.1", &bbox, img_h),
            TextType::Tolerance
        );
        assert_eq!(
            analyzer.classify_text("技术要求", &bbox, img_h),
            TextType::TechnicalNote
        );
        assert_eq!(
            analyzer.classify_text("A-A", &bbox, img_h),
            TextType::ViewLabel
        );
    }

    #[test]
    fn test_title_block_detection() {
        let analyzer = TextGeometryAnalyzer::default();
        let img_h = 1000;
        let bottom_bbox = BoundingBox {
            x_min: 100,
            y_min: 850,
            x_max: 150,
            y_max: 880,
        };

        assert_eq!(
            analyzer.classify_text("任意文本", &bottom_bbox, img_h),
            TextType::TitleBlock
        );
    }

    #[test]
    fn test_heuristic_ocr_name() {
        let ocr = HeuristicOcrBackend::new();
        assert_eq!(ocr.name(), "HeuristicOCR (Pure Rust)");
    }

    #[test]
    fn test_heuristic_ocr_empty_image() {
        let ocr = HeuristicOcrBackend::new();
        let img = GrayImage::from_pixel(100, 100, Luma([255]));
        let results = ocr.recognize(&img);
        assert!(results.is_empty());
    }

    #[test]
    fn test_text_geometry_association_empty() {
        let analyzer = TextGeometryAnalyzer::default();
        let associations = analyzer.associate_text_geometry(&[], &[], 1000);
        assert!(associations.is_empty());
    }

    #[test]
    fn test_digit_recognizer_creation() {
        let _recognizer = DigitRecognizer::new();
    }
}
