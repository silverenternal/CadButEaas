//! Otsu 自适应阈值算法
//!
//! 提供自动阈值计算功能

use image::GrayImage;
use rayon::prelude::*;

/// Otsu 自动阈值计算
///
/// 使用最大类间方差法自动确定最佳阈值
///
/// # 参数
/// - `image`: 输入灰度图像
///
/// # 返回
/// 最佳阈值 (0-255)
pub fn otsu_threshold(image: &GrayImage) -> u8 {
    // 1. 计算直方图
    let mut histogram = [0u32; 256];
    for pixel in image.pixels() {
        histogram[pixel[0] as usize] += 1;
    }

    // 2. 计算归一化直方图
    let total = (image.width() * image.height()) as f64;
    let mut prob = [0.0f64; 256];
    for i in 0..256 {
        prob[i] = histogram[i] as f64 / total;
    }

    // 3. 遍历所有阈值，找到最大类间方差
    let mut max_variance = 0.0f64;
    let mut threshold = 0u8;

    for t in 0..255 {
        let mut w0 = 0.0f64;
        let mut w1 = 0.0f64;
        let mut mean0 = 0.0f64;
        let mut mean1 = 0.0f64;

        // 计算类 0（小于等于阈值）的权重和均值
        for (i, &p) in prob.iter().enumerate().take(t + 1) {
            w0 += p;
            mean0 += i as f64 * p;
        }

        // 计算类 1（大于阈值）的权重和均值
        for (i, &p) in prob.iter().enumerate().skip(t + 1) {
            w1 += p;
            mean1 += i as f64 * p;
        }

        if w0 < 1e-10 || w1 < 1e-10 {
            continue;
        }

        mean0 /= w0;
        mean1 /= w1;

        // 类间方差：w0 * w1 * (mean0 - mean1)²
        let variance = w0 * w1 * (mean0 - mean1).powi(2);

        if variance > max_variance {
            max_variance = variance;
            threshold = t as u8;
        }
    }

    threshold
}

/// 使用 Otsu 阈值的二值化
///
/// # 参数
/// - `image`: 输入灰度图像
///
/// # 返回
/// 二值化后的图像（前景 0，背景 255）
pub fn binary_with_otsu(image: &GrayImage) -> GrayImage {
    let t = otsu_threshold(image);
    threshold_binary(image, t)
}

/// 固定阈值二值化
///
/// # 参数
/// - `image`: 输入灰度图像
/// - `threshold`: 阈值 (0-255)
///
/// # 返回
/// 二值化后的图像（小于阈值为 0，否则为 255）
pub fn threshold_binary(image: &GrayImage, threshold: u8) -> GrayImage {
    let (width, height) = image.dimensions();
    let w = width as usize;

    // Small image fallback
    if width < 100 || height < 100 {
        return threshold_binary_serial(image, threshold);
    }

    let mut result_pixels: Vec<u8> = vec![0; (width * height) as usize];

    result_pixels
        .par_chunks_mut(w)
        .zip(image.as_ref().par_chunks(w))
        .for_each(|(dst_row, src_row)| {
            for (dst, &src) in dst_row.iter_mut().zip(src_row.iter()) {
                *dst = if src < threshold { 0 } else { 255 };
            }
        });

    GrayImage::from_raw(width, height, result_pixels)
        .unwrap_or_else(|| GrayImage::new(width, height))
}

fn threshold_binary_serial(image: &GrayImage, threshold: u8) -> GrayImage {
    let mut result = GrayImage::new(image.width(), image.height());

    for (src, dst) in image.pixels().zip(result.pixels_mut()) {
        dst[0] = if src[0] < threshold { 0 } else { 255 };
    }

    result
}

/// 反向二值化
///
/// # 参数
/// - `image`: 输入灰度图像
/// - `threshold`: 阈值 (0-255)
///
/// # 返回
/// 二值化后的图像（小于阈值为 255，否则为 0）
pub fn threshold_binary_inv(image: &GrayImage, threshold: u8) -> GrayImage {
    let (width, height) = image.dimensions();
    let w = width as usize;

    // Small image fallback
    if width < 100 || height < 100 {
        return threshold_binary_inv_serial(image, threshold);
    }

    let mut result_pixels: Vec<u8> = vec![0; (width * height) as usize];

    result_pixels
        .par_chunks_mut(w)
        .zip(image.as_ref().par_chunks(w))
        .for_each(|(dst_row, src_row)| {
            for (dst, &src) in dst_row.iter_mut().zip(src_row.iter()) {
                *dst = if src < threshold { 255 } else { 0 };
            }
        });

    GrayImage::from_raw(width, height, result_pixels)
        .unwrap_or_else(|| GrayImage::new(width, height))
}

fn threshold_binary_inv_serial(image: &GrayImage, threshold: u8) -> GrayImage {
    let mut result = GrayImage::new(image.width(), image.height());

    for (src, dst) in image.pixels().zip(result.pixels_mut()) {
        dst[0] = if src[0] < threshold { 255 } else { 0 };
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Luma;

    #[test]
    fn test_otsu_threshold_bimodal() {
        // 创建双峰分布的测试图像
        let mut img = GrayImage::new(100, 100);

        // 左半部分暗
        for y in 0..100 {
            for x in 0..50 {
                img.put_pixel(x, y, Luma([50]));
            }
        }

        // 右半部分亮
        for y in 0..100 {
            for x in 50..100 {
                img.put_pixel(x, y, Luma([200]));
            }
        }

        let threshold = otsu_threshold(&img);

        // 阈值应该在两个峰值之间（包括边界）
        // 对于完美的双峰分布，Otsu 阈值应该是 (50+200)/2 = 125
        assert!((50..=200).contains(&threshold), "threshold = {}", threshold);
    }

    #[test]
    fn test_binary_with_otsu() {
        let mut img = GrayImage::new(10, 10);
        for (i, pixel) in img.pixels_mut().enumerate() {
            pixel[0] = (i % 256) as u8;
        }

        let binary = binary_with_otsu(&img);

        // 验证所有像素都是 0 或 255
        for pixel in binary.pixels() {
            assert!(pixel[0] == 0 || pixel[0] == 255);
        }
    }

    #[test]
    fn test_threshold_binary() {
        let mut img = GrayImage::new(2, 2);
        img.put_pixel(0, 0, Luma([100]));
        img.put_pixel(1, 0, Luma([200]));
        img.put_pixel(0, 1, Luma([50]));
        img.put_pixel(1, 1, Luma([150]));

        let binary = threshold_binary(&img, 128);

        assert_eq!(binary.get_pixel(0, 0)[0], 0); // 100 < 128
        assert_eq!(binary.get_pixel(1, 0)[0], 255); // 200 >= 128
        assert_eq!(binary.get_pixel(0, 1)[0], 0); // 50 < 128
        assert_eq!(binary.get_pixel(1, 1)[0], 255); // 150 >= 128
    }
}
