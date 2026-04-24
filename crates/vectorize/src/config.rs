//! 矢量化服务配置

use accelerator_api::{ArcFitConfig, ContourExtractConfig, EdgeDetectConfig, SnapConfig};
use serde::{Deserialize, Serialize};

fn default_max_pixels() -> usize {
    30_000_000
}

/// 矢量化配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorizeConfig {
    /// 二值化阈值 (0-255)
    pub threshold: u8,
    /// 端点吸附容差 (像素)
    pub snap_tolerance_px: f64,
    /// 最小线段长度 (像素)
    pub min_line_length_px: f64,
    /// 最大角度偏差 (度)
    pub max_angle_dev_deg: f64,
    /// 是否进行骨架化
    pub skeletonize: bool,
    /// 是否使用 OpenCV 加速（需要 `opencv` feature）
    pub use_opencv: bool,
    /// 是否使用自适应阈值（Otsu）
    pub adaptive_threshold: bool,
    /// 是否使用霍夫变换辅助直线检测
    pub use_hough: bool,
    /// 图像预处理选项
    pub preprocessing: PreprocessingConfig,
    /// 线型识别
    pub line_type_detection: bool,
    /// 圆弧拟合
    pub arc_fitting: bool,
    /// 断点连接
    pub gap_filling: bool,
    /// 质量评估
    pub quality_assessment: bool,
    /// 文字标注分离（检测并过滤文字连通区域）
    pub text_separation: bool,
    /// DPI 自适应（如果为 true，则根据 dpi 自动调整像素阈值）
    pub dpi_adaptive: bool,
    /// 参考 DPI（用于计算缩放比例）
    pub reference_dpi: f64,
    /// DPI 自适应参数缩放系数（用于调整敏感度，默认为 1.0）
    pub dpi_scale_factor: f64,
    /// OpenCV 多边形简化精度（epsilon 值，仅在使用 OpenCV 时有效）
    pub opencv_approx_epsilon: Option<f64>,
    /// 最大图像像素数限制（默认 30,000,000，约 5477x5477，支持 A3 @ 300 DPI）
    /// 超过此限制的图像会自动缩放（保持宽高比）
    #[serde(default = "default_max_pixels")]
    pub max_pixels: usize,
    /// 是否使用加速器进行边缘检测（GPU 加速，如果可用）
    pub use_accelerator_edge_detect: bool,
    /// 自动纸张检测与裁剪（去除背景黑边/干扰）
    pub auto_crop_paper: bool,
    /// 透视变换校正（对拍照斜视图纸进行校正）
    pub perspective_correction: bool,
    /// 霍夫变换辅助缺口填充（处理更大间隔的断点）
    pub hough_gap_filling: bool,
    /// 霍夫变换直线检测阈值（投票数，越大检测越少但越准确）
    pub hough_threshold: u32,
    /// 建筑规则几何校正（正交性/平行性校正，针对建筑图纸）
    pub architectural_correction: bool,
    /// 自适应参数调整（根据图像质量自动调整预处理参数）
    pub adaptive_params: bool,
}

/// 图像预处理配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreprocessingConfig {
    /// 是否启用去噪
    pub denoise: bool,
    /// 去噪方法 ("median", "gaussian", "none")
    pub denoise_method: String,
    /// 去噪参数
    pub denoise_strength: f32,
    /// 是否增强对比度
    pub enhance_contrast: bool,
    /// CLAHE 参数
    pub clahe_clip_limit: f32,
    pub clahe_tile_size: u32,
}

impl Default for PreprocessingConfig {
    fn default() -> Self {
        Self {
            denoise: true,
            denoise_method: "median".to_string(),
            denoise_strength: 3.0,
            enhance_contrast: true,
            clahe_clip_limit: 2.5,
            clahe_tile_size: 8,
        }
    }
}

impl Default for VectorizeConfig {
    fn default() -> Self {
        Self {
            threshold: 128,
            snap_tolerance_px: 2.0,
            min_line_length_px: 10.0,
            max_angle_dev_deg: 5.0,
            skeletonize: true,
            #[cfg(feature = "opencv")]
            use_opencv: true,
            #[cfg(not(feature = "opencv"))]
            use_opencv: false,
            adaptive_threshold: true,
            use_hough: false,
            preprocessing: PreprocessingConfig::default(),
            line_type_detection: true,
            arc_fitting: false,
            gap_filling: true,
            quality_assessment: true,
            text_separation: true,
            dpi_adaptive: true,
            reference_dpi: 300.0,
            dpi_scale_factor: 1.0,
            opencv_approx_epsilon: Some(2.0),
            max_pixels: 30_000_000,
            use_accelerator_edge_detect: true,
            auto_crop_paper: true,
            perspective_correction: true,
            hough_gap_filling: true,
            hough_threshold: 50,
            architectural_correction: true,
            adaptive_params: true,
        }
    }
}

impl VectorizeConfig {
    /// 创建边缘检测配置
    pub fn to_edge_detect_config(&self) -> EdgeDetectConfig {
        EdgeDetectConfig {
            low_threshold: self.threshold as f64,
            high_threshold: (self.threshold as f64) * 2.0,
            sobel_kernel_size: 3,
            adaptive_threshold: self.adaptive_threshold,
        }
    }

    /// 创建轮廓提取配置
    pub fn to_contour_extract_config(&self) -> ContourExtractConfig {
        ContourExtractConfig {
            min_contour_length: self.min_line_length_px as usize,
            simplify_epsilon: self.snap_tolerance_px,
            simplify: true,
        }
    }

    /// 创建圆弧拟合配置
    pub fn to_arc_fit_config(&self) -> ArcFitConfig {
        ArcFitConfig {
            max_error: self.snap_tolerance_px,
            min_points: 3,
        }
    }

    /// 创建端点吸附配置
    pub fn to_snap_config(&self) -> SnapConfig {
        SnapConfig {
            tolerance: self.snap_tolerance_px,
            use_rtree: true,
        }
    }
}
