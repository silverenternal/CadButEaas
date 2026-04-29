//! 光栅图像预处理
//!
//! 针对扫描图纸的预处理：阴影去除、对比度增强、亮度归一化等。
//! 专门针对工程图纸场景优化，保留线条细节。

use image::{DynamicImage, GrayImage, Luma};

/// 预处理配置
#[derive(Debug, Clone)]
pub struct PreprocessConfig {
    /// 是否自动调整对比度
    pub auto_contrast: bool,
    /// 对比度裁剪百分比（0.0-5.0），值越大对比度越强
    pub contrast_saturation: f64,
    /// 是否去除阴影（适用于不均匀光照的扫描件）
    pub remove_shadow: bool,
    /// 阴影去除半径（像素，3-11）
    pub shadow_radius: u32,
    /// 是否锐化边缘
    pub sharpen: bool,
    /// 锐化强度（0.5-2.0）
    pub sharpen_amount: f32,
    /// 是否中值滤波去噪
    pub denoise: bool,
    /// 中值滤波核大小（3/5/7）
    pub denoise_kernel_size: u32,
    /// 是否自动二值化（Otsu）
    pub auto_threshold: bool,
}

impl Default for PreprocessConfig {
    fn default() -> Self {
        Self {
            auto_contrast: true,
            contrast_saturation: 1.0,
            remove_shadow: true,
            shadow_radius: 7,
            sharpen: true,
            sharpen_amount: 1.0,
            denoise: true,
            denoise_kernel_size: 3,
            auto_threshold: true,
        }
    }
}

impl PreprocessConfig {
    /// 快速配置：高质量扫描件
    pub fn clean_scan() -> Self {
        Self {
            auto_contrast: true,
            contrast_saturation: 0.5,
            remove_shadow: false,
            shadow_radius: 5,
            sharpen: true,
            sharpen_amount: 0.7,
            denoise: false,
            denoise_kernel_size: 3,
            auto_threshold: true,
        }
    }

    /// 快速配置：低质量扫描件（阴影、折痕、褪色）
    pub fn poor_scan() -> Self {
        Self {
            auto_contrast: true,
            contrast_saturation: 2.0,
            remove_shadow: true,
            shadow_radius: 9,
            sharpen: true,
            sharpen_amount: 1.5,
            denoise: true,
            denoise_kernel_size: 5,
            auto_threshold: true,
        }
    }

    /// 快速配置：照片拍摄的图纸
    pub fn photo_capture() -> Self {
        Self {
            auto_contrast: true,
            contrast_saturation: 2.5,
            remove_shadow: true,
            shadow_radius: 11,
            sharpen: true,
            sharpen_amount: 2.0,
            denoise: true,
            denoise_kernel_size: 5,
            auto_threshold: true,
        }
    }
}

/// 光栅图像预处理器
pub struct RasterPreprocessor {
    config: PreprocessConfig,
}

impl Default for RasterPreprocessor {
    fn default() -> Self {
        Self::new(PreprocessConfig::default())
    }
}

impl RasterPreprocessor {
    /// 创建预处理器
    pub fn new(config: PreprocessConfig) -> Self {
        Self { config }
    }

    /// 执行完整预处理流程
    pub fn process(&self, img: &DynamicImage) -> GrayImage {
        let mut gray = img.to_luma8();

        // 步骤 1: 阴影去除
        if self.config.remove_shadow {
            gray = self.remove_shadow(&gray);
        }

        // 步骤 2: 中值滤波去噪
        if self.config.denoise {
            gray = self.median_filter(&gray, self.config.denoise_kernel_size);
        }

        // 步骤 3: 对比度增强
        if self.config.auto_contrast {
            gray = self.autocontrast(&gray, self.config.contrast_saturation);
        }

        // 步骤 4: 边缘锐化
        if self.config.sharpen {
            gray = self.sharpen(&gray, self.config.sharpen_amount);
        }

        // 步骤 5: 二值化
        if self.config.auto_threshold {
            gray = self.otsu_threshold(&gray);
        }

        gray
    }

    /// 阴影去除：使用形态学开运算估计背景
    /// 原理：对图像做最大值滤波得到背景估计，然后用原图/背景得到归一化
    fn remove_shadow(&self, gray: &GrayImage) -> GrayImage {
        let (width, height) = gray.dimensions();
        let radius = self.config.shadow_radius;

        // 步骤 1: 最大值滤波（膨胀）估计背景
        let background = self.max_filter(gray, radius);

        // 步骤 2: 逐像素归一化（消除光照不均）
        let mut result = GrayImage::new(width, height);
        for y in 0..height {
            for x in 0..width {
                let original = gray.get_pixel(x, y)[0] as f32;
                let bg = background.get_pixel(x, y)[0] as f32;
                let normalized = if bg > 0.0 {
                    (original / bg * 255.0).clamp(0.0, 255.0) as u8
                } else {
                    original as u8
                };
                result.put_pixel(x, y, Luma([normalized]));
            }
        }

        result
    }

    /// 最大值滤波（用于背景估计）
    fn max_filter(&self, gray: &GrayImage, radius: u32) -> GrayImage {
        let (width, height) = gray.dimensions();
        let mut result = GrayImage::new(width, height);

        for y in 0..height {
            for x in 0..width {
                let mut max_val = 0u8;
                let start_y = y.saturating_sub(radius);
                let end_y = (y + radius).min(height - 1);
                let start_x = x.saturating_sub(radius);
                let end_x = (x + radius).min(width - 1);

                for yy in start_y..=end_y {
                    for xx in start_x..=end_x {
                        let val = gray.get_pixel(xx, yy)[0];
                        if val > max_val {
                            max_val = val;
                        }
                    }
                }

                result.put_pixel(x, y, Luma([max_val]));
            }
        }

        result
    }

    /// 中值滤波去噪（椒盐噪声）
    fn median_filter(&self, gray: &GrayImage, kernel_size: u32) -> GrayImage {
        let (width, height) = gray.dimensions();
        let mut result = GrayImage::new(width, height);
        let half = kernel_size / 2;

        for y in 0..height {
            for x in 0..width {
                let mut pixels = Vec::with_capacity((kernel_size * kernel_size) as usize);
                let start_y = y.saturating_sub(half);
                let end_y = (y + half).min(height - 1);
                let start_x = x.saturating_sub(half);
                let end_x = (x + half).min(width - 1);

                for yy in start_y..=end_y {
                    for xx in start_x..=end_x {
                        pixels.push(gray.get_pixel(xx, yy)[0]);
                    }
                }

                pixels.sort();
                let median = pixels[pixels.len() / 2];
                result.put_pixel(x, y, Luma([median]));
            }
        }

        result
    }

    /// 自动对比度调整：基于直方图饱和裁剪
    fn autocontrast(&self, gray: &GrayImage, saturation_percent: f64) -> GrayImage {
        let (width, height) = gray.dimensions();

        // 统计直方图
        let mut histogram = [0usize; 256];
        for pixel in gray.pixels() {
            histogram[pixel[0] as usize] += 1;
        }

        let total_pixels = (width * height) as f64;
        let cutoff = (total_pixels * saturation_percent / 200.0).round() as usize;

        // 找到低点和高点
        let mut cumulative = 0;
        let mut low = 0u8;
        for (i, &count) in histogram.iter().enumerate() {
            cumulative += count;
            if cumulative > cutoff {
                low = i as u8;
                break;
            }
        }

        cumulative = 0;
        let mut high = 255u8;
        for (i, &count) in histogram.iter().enumerate().rev() {
            cumulative += count;
            if cumulative > cutoff {
                high = i as u8;
                break;
            }
        }

        // 边界情况
        if low >= high {
            return gray.clone();
        }

        // 线性拉伸
        let scale = 255.0 / (high as f32 - low as f32);
        let mut result = GrayImage::new(width, height);
        for y in 0..height {
            for x in 0..width {
                let val = gray.get_pixel(x, y)[0];
                let stretched = ((val as f32 - low as f32) * scale).clamp(0.0, 255.0) as u8;
                result.put_pixel(x, y, Luma([stretched]));
            }
        }

        result
    }

    /// 边缘锐化：3x3 拉普拉斯算子
    fn sharpen(&self, gray: &GrayImage, amount: f32) -> GrayImage {
        let (width, height) = gray.dimensions();
        let mut result = GrayImage::new(width, height);

        // 3x3 拉普拉斯锐化核
        let kernel = [[0.0, -1.0, 0.0], [-1.0, 5.0, -1.0], [0.0, -1.0, 0.0]];

        // 乘以强度因子
        let scaled_kernel: [[f32; 3]; 3] = kernel.map(|row| {
            row.map(|v| {
                if v == 5.0 {
                    1.0 + (v - 1.0) * amount
                } else {
                    v * amount
                }
            })
        });

        for y in 0..height {
            for x in 0..width {
                if x == 0 || x == width - 1 || y == 0 || y == height - 1 {
                    // 边界复制原值
                    result.put_pixel(x, y, Luma([gray.get_pixel(x, y)[0]]));
                } else {
                    let mut sum = 0.0f32;
                    for ky in 0..3 {
                        for kx in 0..3 {
                            let px = gray.get_pixel(x + kx - 1, y + ky - 1)[0] as f32;
                            sum += px * scaled_kernel[ky as usize][kx as usize];
                        }
                    }
                    let val = sum.clamp(0.0, 255.0) as u8;
                    result.put_pixel(x, y, Luma([val]));
                }
            }
        }

        result
    }

    /// Otsu 自动阈值二值化
    /// 自动找到前景（线条）和背景的最佳分割阈值
    #[allow(clippy::needless_range_loop)] // 0..256 阈值迭代是算法固有特性
    pub fn otsu_threshold(&self, gray: &GrayImage) -> GrayImage {
        let (width, height) = gray.dimensions();

        // 统计直方图
        let mut histogram = [0usize; 256];
        for pixel in gray.pixels() {
            histogram[pixel[0] as usize] += 1;
        }

        let total = (width * height) as f64;

        // Otsu 算法：最大化类间方差
        let mut best_threshold = 127u8;
        let mut max_variance = 0.0f64;

        let mut sum_total = 0.0f64;
        for t in 0..256 {
            sum_total += t as f64 * histogram[t] as f64;
        }

        let mut sum_b = 0.0f64;
        let mut weight_b = 0.0f64;
        let mut weight_f;

        for t in 0..256 {
            weight_b += histogram[t] as f64;
            if weight_b == 0.0 {
                continue;
            }

            weight_f = total - weight_b;
            if weight_f == 0.0 {
                break;
            }

            sum_b += t as f64 * histogram[t] as f64;
            let mean_b = sum_b / weight_b;
            let mean_f = (sum_total - sum_b) / weight_f;

            // 类间方差
            let between = weight_b * weight_f * (mean_b - mean_f) * (mean_b - mean_f);

            if between > max_variance {
                max_variance = between;
                best_threshold = t as u8;
            }
        }

        // 应用阈值
        let mut result = GrayImage::new(width, height);
        for y in 0..height {
            for x in 0..width {
                let val = gray.get_pixel(x, y)[0];
                let binary = if val <= best_threshold { 0 } else { 255 };
                result.put_pixel(x, y, Luma([binary]));
            }
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{GrayImage, Luma};

    #[test]
    fn test_preprocess_config_defaults() {
        let config = PreprocessConfig::default();
        assert!(config.auto_contrast);
        assert!(config.remove_shadow);
        assert!(config.sharpen);
        assert!(config.denoise);
        assert!(config.auto_threshold);
    }

    #[test]
    fn test_preprocess_config_presets() {
        let clean = PreprocessConfig::clean_scan();
        assert!(!clean.remove_shadow);
        assert!(!clean.denoise);

        let poor = PreprocessConfig::poor_scan();
        assert!(poor.remove_shadow);
        assert!(poor.denoise);
        assert_eq!(poor.denoise_kernel_size, 5);
    }

    #[test]
    fn test_otsu_threshold_simple() {
        // 创建简单的测试图像：10x10，一半0，一半255
        let mut gray = GrayImage::new(10, 10);
        for y in 0..10 {
            for x in 0..10 {
                let val = if x < 5 { 0 } else { 255 };
                gray.put_pixel(x, y, Luma([val as u8]));
            }
        }

        let preprocessor = RasterPreprocessor::default();
        let result = preprocessor.otsu_threshold(&gray);

        // 阈值应该在中间，二值化后保持不变
        for y in 0..10 {
            for x in 0..10 {
                let expected = if x < 5 { 0 } else { 255 };
                assert_eq!(result.get_pixel(x, y)[0], expected);
            }
        }
    }

    #[test]
    fn test_autocontrast() {
        // 创建对比度低的图像（范围 100-150）
        let mut gray = GrayImage::new(10, 10);
        for y in 0..10 {
            for x in 0..10 {
                let val = 100 + x * 5;
                gray.put_pixel(x, y, Luma([val as u8]));
            }
        }

        let preprocessor = RasterPreprocessor::default();
        let result = preprocessor.autocontrast(&gray, 1.0);

        // 验证对比度被拉伸
        let mut min_val = 255;
        let mut max_val = 0;
        for pixel in result.pixels() {
            min_val = min_val.min(pixel[0]);
            max_val = max_val.max(pixel[0]);
        }
        assert_eq!(min_val, 0);
        assert_eq!(max_val, 255);
    }
}
