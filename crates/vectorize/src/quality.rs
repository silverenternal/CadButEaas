//! 图像质量评估模块

use image::GrayImage;

/// 评估图像质量评分 (0-100)
///
/// # 评估维度
/// 1. 对比度 - 检查图像是否有足够的黑白对比
/// 2. 噪声水平 - 检查图像是否有过多噪点
/// 3. 边缘清晰度 - 检查边缘是否连续清晰
/// 4. 亮度分布 - 检查是否有阴影或光照不均
pub fn evaluate_image_quality(image: &GrayImage) -> f64 {
    let (width, height) = image.dimensions();
    let pixels = image.as_raw();

    // 1. 计算亮度直方图
    let mut histogram = [0usize; 256];
    for &pixel in pixels {
        histogram[pixel as usize] += 1;
    }

    // 2. 对比度评分（0-25）- 使用标准差
    let total_pixels = (width * height) as usize;
    let mean_brightness = (0..256)
        .map(|i| i as f64 * histogram[i] as f64)
        .sum::<f64>() / total_pixels as f64;

    let variance = (0..256)
        .map(|i| {
            let diff = i as f64 - mean_brightness;
            diff * diff * histogram[i] as f64
        })
        .sum::<f64>() / total_pixels as f64;

    let std_dev = variance.sqrt();
    let contrast_score = (std_dev / 128.0 * 25.0).min(25.0);

    // 3. 噪声评分（0-25）
    let mut noise_score = 25.0;
    let mut noise_count = 0;
    for y in 1..(height - 1) {
        for x in 1..(width - 1) {
            let center = image.get_pixel(x, y)[0] as i16;
            let neighbors = [
                image.get_pixel(x - 1, y)[0] as i16,
                image.get_pixel(x + 1, y)[0] as i16,
                image.get_pixel(x, y - 1)[0] as i16,
                image.get_pixel(x, y + 1)[0] as i16,
            ];
            let avg_neighbor = neighbors.iter().sum::<i16>() / 4;

            if (center - avg_neighbor).abs() > 50 {
                noise_count += 1;
            }
        }
    }
    let noise_ratio = noise_count as f64 / (width * height) as f64;
    if noise_ratio > 0.1 {
        noise_score = (25.0 * (1.0 - noise_ratio * 5.0)).max(0.0);
    }

    // 4. 边缘清晰度评分（0-25）
    let mut edge_score = 25.0;
    let mut strong_edges = 0;
    let mut weak_edges = 0;
    for y in 1..(height - 1) {
        for x in 1..(width - 1) {
            let gx = (image.get_pixel(x + 1, y - 1)[0] as i32
                + 2i32.saturating_mul(image.get_pixel(x + 1, y)[0] as i32)
                + image.get_pixel(x + 1, y + 1)[0] as i32)
                - (image.get_pixel(x - 1, y - 1)[0] as i32
                + 2i32.saturating_mul(image.get_pixel(x - 1, y)[0] as i32)
                + image.get_pixel(x - 1, y + 1)[0] as i32);

            let gy = (image.get_pixel(x - 1, y + 1)[0] as i32
                + 2i32.saturating_mul(image.get_pixel(x, y + 1)[0] as i32)
                + image.get_pixel(x + 1, y + 1)[0] as i32)
                - (image.get_pixel(x - 1, y - 1)[0] as i32
                + 2i32.saturating_mul(image.get_pixel(x, y - 1)[0] as i32)
                + image.get_pixel(x + 1, y - 1)[0] as i32);

            let magnitude = (gx * gx + gy * gy) as f64;

            if magnitude > 10000.0 {
                strong_edges += 1;
            } else if magnitude > 2500.0 {
                weak_edges += 1;
            }
        }
    }
    let edge_ratio = strong_edges as f64 / (strong_edges + weak_edges + 1) as f64;
    if edge_ratio < 0.3 {
        edge_score = edge_ratio * 25.0;
    }

    // 5. 亮度分布评分（0-25）
    let brightness_score = {
        let dark_pixels: usize = histogram[0..80].iter().sum();
        let bright_pixels: usize = histogram[180..256].iter().sum();

        let bimodality = (dark_pixels + bright_pixels) as f64 / total_pixels as f64;
        if bimodality > 0.7 {
            25.0
        } else if bimodality > 0.5 {
            20.0
        } else if bimodality > 0.3 {
            15.0
        } else {
            (bimodality * 50.0).min(25.0)
        }
    };

    // 总分
    let total_score = contrast_score + noise_score + edge_score + brightness_score;
    total_score.clamp(0.0, 100.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Luma;

    #[test]
    fn test_quality_high_contrast() {
        let mut img = GrayImage::new(100, 100);
        // 创建高对比度图像（一半黑，一半白）
        for y in 0..100 {
            for x in 0..100 {
                if x < 50 {
                    img.put_pixel(x, y, Luma([0]));
                } else {
                    img.put_pixel(x, y, Luma([255]));
                }
            }
        }

        let score = evaluate_image_quality(&img);
        assert!(score > 70.0); // 高对比度应该得分高
    }

    #[test]
    fn test_quality_low_contrast() {
        let mut img = GrayImage::new(100, 100);
        // 创建低对比度图像（全部灰色）
        for y in 0..100 {
            for x in 0..100 {
                img.put_pixel(x, y, Luma([128]));
            }
        }

        let score = evaluate_image_quality(&img);
        assert!(score < 50.0); // 低对比度应该得分低
    }
}
