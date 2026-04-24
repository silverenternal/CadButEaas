//! 自适应参数调整
//!
//! 根据图像质量评分和分辨率自动推导最佳预处理参数，
//! 降低用户调参负担，提高对不同质量图纸的鲁棒性。

use crate::config::VectorizeConfig;

/// 自适应参数计算 - 根据图像质量自动调整参数
///
/// # 参数
/// - `quality_score`: 图像质量评分 (0-100)，越高质量越好
/// - `width`: 图像宽度像素
/// - `height`: 图像高度像素
/// - `base_config`: 基础配置，作为调整起点
///
/// # 返回
/// 调整后的配置
pub fn adapt_parameters(
    quality_score: f64,
    width: u32,
    height: u32,
    base_config: &VectorizeConfig,
) -> VectorizeConfig {
    let mut result = base_config.clone();

    // 根据质量评分调整预处理参数
    // 质量越低，越需要更强的预处理
    if quality_score < 60.0 {
        // 低质量图像 - 增强去噪和对比度
        result.preprocessing.denoise = true;
        result.preprocessing.enhance_contrast = true;

        // 质量越低，去噪强度越高
        // 60 → 3.0, 30 → 6.0, 0 → 8.0
        let denoise_strength = (3.0 + (60.0 - quality_score) * (5.0 / 60.0)).clamp(3.0, 8.0);
        result.preprocessing.denoise_strength = denoise_strength as f32;

        // 低质量图像使用更强的对比度限制
        result.preprocessing.clahe_clip_limit = 3.5;

        // 低质量图像更激进的文字分离（更多噪声被过滤）
        result.text_separation = true;

        // 低质量图像启用更强的缺口填充
        result.hough_gap_filling = true;
    } else if quality_score < 80.0 {
        // 中等质量 - 默认参数但适度增强
        result.preprocessing.denoise = true;
        result.preprocessing.enhance_contrast = true;
        result.preprocessing.denoise_strength = 3.5;
        result.preprocessing.clahe_clip_limit = 3.0;
    } else {
        // 高质量 - 最小预处理，保留细节
        result.preprocessing.denoise = true;
        result.preprocessing.enhance_contrast = false;
        result.preprocessing.denoise_strength = 1.5;
    }

    // 根据分辨率自适应调整像素容差
    // 大图像需要更大的容差，因为像素对应更小的实际尺寸
    let total_pixels = (width * height) as f64;
    let log_pixels = total_pixels.log10();

    // 基准是 ~1000x1000 = 1e6 像素，log10 = 6
    // 根据对数缩放调整容差
    let scale_factor = if log_pixels > 6.0 {
        // 大图像放大容差
        (log_pixels - 6.0) * 0.2 + 1.0
    } else if log_pixels < 5.0 {
        // 小图像缩小容差
        (log_pixels - 5.0) * 0.2 + 1.0
    } else {
        1.0
    };

    // 应用缩放
    result.snap_tolerance_px *= scale_factor;
    result.min_line_length_px *= scale_factor;

    // 霍夫阈值也随分辨率调整
    // 大图像需要更高阈值，减少误检
    result.hough_threshold = ((result.hough_threshold as f64) * scale_factor.sqrt()).round() as u32;

    result
}

/// 根据 DPI 计算实际像素缩放
///
/// 当输入图像带有 DPI 信息时，根据参考 DPI 调整参数
pub fn dpi_adjust_parameters(
    dpi: f64,
    reference_dpi: f64,
    scale_factor: f64,
    config: &mut VectorizeConfig,
) {
    if dpi <= 0.0 || reference_dpi <= 0.0 {
        return;
    }

    // 比例 = (实际 DPI) / (参考 DPI) * scale_factor
    let ratio = (dpi / reference_dpi) * scale_factor;

    // 所有像素相关参数按比例缩放
    config.snap_tolerance_px *= ratio;
    config.min_line_length_px *= ratio;
    config.hough_threshold = ((config.hough_threshold as f64) * ratio.sqrt()).round() as u32;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::VectorizeConfig;

    #[test]
    fn test_adapt_high_quality() {
        let base = VectorizeConfig::default();
        let adapted = adapt_parameters(90.0, 1000, 1000, &base);

        // 高质量应该降低去噪强度，关闭对比度增强
        assert!(adapted.preprocessing.denoise_strength < 3.0);
        assert!(!adapted.preprocessing.enhance_contrast);
    }

    #[test]
    fn test_adapt_low_quality() {
        let base = VectorizeConfig::default();
        let adapted = adapt_parameters(40.0, 1000, 1000, &base);

        // 低质量应该增强去噪，开启对比度增强
        assert!(adapted.preprocessing.denoise_strength > 3.0);
        assert!(adapted.preprocessing.enhance_contrast);
        assert!(adapted.text_separation);
        assert!(adapted.hough_gap_filling);
    }

    #[test]
    fn test_adapt_large_resolution() {
        let base = VectorizeConfig::default();
        let base_tolerance = base.snap_tolerance_px;
        let adapted = adapt_parameters(70.0, 4000, 3000, &base);

        // 大图像应该放大容差
        assert!(adapted.snap_tolerance_px > base_tolerance);
    }

    #[test]
    fn test_dpi_adjust() {
        let mut config = VectorizeConfig::default();
        let original_tolerance = config.snap_tolerance_px;

        // 更高 DPI = 像素更小 → 容差应该放大
        dpi_adjust_parameters(600.0, 300.0, 1.0, &mut config);
        assert!(config.snap_tolerance_px > original_tolerance);

        // 更低 DPI = 像素更大 → 容差应该缩小
        let mut config2 = VectorizeConfig::default();
        dpi_adjust_parameters(150.0, 300.0, 1.0, &mut config2);
        assert!(config2.snap_tolerance_px < original_tolerance);
    }
}
