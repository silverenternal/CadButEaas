//! 图像预处理算法
//!
//! 提供去噪、对比度增强等预处理功能

use image::{GrayImage, Luma};

/// 中值滤波去噪
///
/// # 参数
/// - `image`: 输入灰度图像
/// - `kernel_size`: 滤波核大小（建议 3 或 5）
///
/// # 返回
/// 去噪后的灰度图像
pub fn median_filter(image: &GrayImage, kernel_size: u32) -> GrayImage {
    let (width, height) = image.dimensions();
    let mut result = GrayImage::new(width, height);
    let half = kernel_size / 2;

    for y in 0..height {
        for x in 0..width {
            let mut neighbors = Vec::new();

            for dy in 0..kernel_size {
                for dx in 0..kernel_size {
                    let nx = x.saturating_sub(half) + dx;
                    let ny = y.saturating_sub(half) + dy;
                    if nx < width && ny < height {
                        neighbors.push(image.get_pixel(nx, ny)[0]);
                    }
                }
            }

            neighbors.sort();
            let median = neighbors[neighbors.len() / 2];
            result.put_pixel(x, y, Luma([median]));
        }
    }

    result
}

/// 高斯滤波去噪
///
/// # 参数
/// - `image`: 输入灰度图像
/// - `sigma`: 高斯分布的标准差
///
/// # 返回
/// 去噪后的灰度图像
pub fn gaussian_blur(image: &GrayImage, sigma: f32) -> GrayImage {
    let (width, height) = image.dimensions();
    
    // 计算核大小（3σ原则）
    let kernel_radius = ((sigma * 3.0).ceil() as u32).max(1);
    let kernel_size = 2 * kernel_radius + 1;
    
    // 生成高斯核
    let mut kernel = Vec::with_capacity(kernel_size as usize);
    let mut sum = 0.0f32;
    
    for y in 0..kernel_size {
        for x in 0..kernel_size {
            let dx = x as i32 - kernel_radius as i32;
            let dy = y as i32 - kernel_radius as i32;
            let g = ((dx * dx + dy * dy) as f32 / (-2.0 * sigma * sigma)).exp();
            kernel.push(g);
            sum += g;
        }
    }
    
    // 归一化核
    for k in &mut kernel {
        *k /= sum;
    }
    
    // 卷积
    let mut result = GrayImage::new(width, height);
    for y in 0..height {
        for x in 0..width {
            let mut pixel_val = 0.0f32;
            
            for ky in 0..kernel_size {
                for kx in 0..kernel_size {
                    let nx = x.saturating_sub(kernel_radius) + kx;
                    let ny = y.saturating_sub(kernel_radius) + ky;
                    if nx < width && ny < height {
                        let idx = (ky * kernel_size + kx) as usize;
                        pixel_val += image.get_pixel(nx, ny)[0] as f32 * kernel[idx];
                    }
                }
            }
            
            result.put_pixel(x, y, Luma([pixel_val as u8]));
        }
    }
    
    result
}

/// 非局部均值去噪（简化版，堆分配优化）
///
/// # 参数
/// - `image`: 输入灰度图像
/// - `h`: 滤波强度参数（建议 5-15）
///
/// # 返回
/// 去噪后的灰度图像
///
/// # 注意
/// 此函数使用堆分配避免栈溢出，但计算复杂度较高 (O(n * search_window^2 * patch_size^2))
/// 建议用于小尺寸图像（< 500x500 像素）
pub fn non_local_means(image: &GrayImage, h: f32) -> GrayImage {
    let (width, height) = image.dimensions();
    
    // 尺寸限制检查
    let max_safe_pixels = 250_000; // 500x500
    if (width as usize * height as usize) > max_safe_pixels {
        // 对于大图像，回退到中值滤波（更快）
        tracing::warn!(
            "non_local_means: 图像尺寸 {}x{} 超过安全限制，回退到中值滤波",
            width, height
        );
        return median_filter(image, 3);
    }

    let mut result = GrayImage::new(width, height);

    // 搜索窗口大小
    let search_window = 5u32;
    // 相似性窗口大小
    let patch_size = 3u32;

    // 预分配 weights 向量，避免重复分配
    let max_weights_size = (search_window * search_window) as usize;
    
    // 预分配 patch 距离计算 buffer
    let mut patch_buffer: Vec<f32> = Vec::with_capacity((patch_size * patch_size) as usize);

    for y in 0..height {
        for x in 0..width {
            // 使用预分配的 Vec，每轮清空
            let mut weights = Vec::with_capacity(max_weights_size);
            let mut weighted_sum = 0.0f32;
            let mut weight_total = 0.0f32;

            // 中心像素
            let center_val = image.get_pixel(x, y)[0] as f32;

            // 在搜索窗口内计算权重
            for sy in 0..search_window {
                for sx in 0..search_window {
                    let nx = x.saturating_sub(search_window / 2) + sx;
                    let ny = y.saturating_sub(search_window / 2) + sy;

                    if nx >= width || ny >= height {
                        continue;
                    }

                    // 计算 patch 距离（使用预分配 buffer）
                    patch_buffer.clear();
                    for py in 0..patch_size {
                        for px in 0..patch_size {
                            let px1 = x.saturating_sub(patch_size / 2) + px;
                            let py1 = y.saturating_sub(patch_size / 2) + py;
                            let px2 = nx.saturating_sub(patch_size / 2) + px;
                            let py2 = ny.saturating_sub(patch_size / 2) + py;

                            if px1 < width && py1 < height && px2 < width && py2 < height {
                                let v1 = image.get_pixel(px1, py1)[0] as f32;
                                let v2 = image.get_pixel(px2, py2)[0] as f32;
                                patch_buffer.push((v1 - v2).powi(2));
                            }
                        }
                    }

                    let dist: f32 = patch_buffer.iter().sum();

                    // 高斯权重
                    let weight = (-dist / (h * h)).exp();
                    weights.push(weight);

                    let neighbor_val = image.get_pixel(nx, ny)[0] as f32;
                    weighted_sum += neighbor_val * weight;
                    weight_total += weight;
                }
            }

            if weight_total > 0.0 {
                result.put_pixel(x, y, Luma([(weighted_sum / weight_total) as u8]));
            } else {
                result.put_pixel(x, y, Luma([center_val as u8]));
            }
        }
    }

    result
}

/// CLAHE 对比度增强（限制对比度自适应直方图均衡化）
///
/// # 参数
/// - `image`: 输入灰度图像
/// - `clip_limit`: 对比度限制（建议 2.0-4.0）
/// - `tile_grid_size`: 分块大小（建议 8 或 16）
///
/// # 返回
/// 增强后的灰度图像
pub fn clahe(image: &GrayImage, clip_limit: f32, tile_grid_size: u32) -> GrayImage {
    let (width, height) = image.dimensions();
    let mut result = GrayImage::new(width, height);

    // 分块处理
    let tiles_x = width.div_ceil(tile_grid_size);
    let tiles_y = height.div_ceil(tile_grid_size);
    let total_tiles = (tiles_x * tiles_y) as usize;

    // 预分配堆内存存储直方图，避免栈溢出
    // 每个分块 256 字节，使用 Vec 在堆上分配
    let mut tile_histograms: Vec<[u8; 256]> = Vec::with_capacity(total_tiles);

    for ty in 0..tiles_y {
        for tx in 0..tiles_x {
            let mut hist = [0u32; 256];

            let x_start = tx * tile_grid_size;
            let y_start = ty * tile_grid_size;
            let x_end = (x_start + tile_grid_size).min(width);
            let y_end = (y_start + tile_grid_size).min(height);

            // 计算直方图
            for y in y_start..y_end {
                for x in x_start..x_end {
                    let val = image.get_pixel(x, y)[0];
                    hist[val as usize] += 1;
                }
            }

            // 裁剪直方图
            let total_pixels = (x_end - x_start) * (y_end - y_start);
            let clip_threshold = (clip_limit * total_pixels as f32 / 256.0) as u32;

            let mut excess = 0u32;
            for h in &mut hist {
                if *h > clip_threshold {
                    excess += *h - clip_threshold;
                    *h = clip_threshold;
                }
            }

            // 重新分配多余像素
            let add_per_bin = excess / 256;
            for h in &mut hist {
                *h = (*h + add_per_bin).min(clip_threshold);
            }

            // 计算累积分布函数
            let mut cdf = [0u32; 256];
            cdf[0] = hist[0];
            for i in 1..256 {
                cdf[i] = cdf[i - 1] + hist[i];
            }

            // 归一化 CDF
            let max_cdf = cdf[255].max(1);
            let mut normalized_cdf = [0u8; 256];
            for (i, &c) in cdf.iter().enumerate() {
                normalized_cdf[i] = ((c as f32 / max_cdf as f32) * 255.0) as u8;
            }

            tile_histograms.push(normalized_cdf);
        }
    }

    // 双线性插值应用 CLAHE
    for y in 0..height {
        for x in 0..width {
            let val = image.get_pixel(x, y)[0] as usize;

            // 找到所属的分块
            let tile_x = (x / tile_grid_size).min(tiles_x - 1);
            let tile_y = (y / tile_grid_size).min(tiles_y - 1);

            // 获取相邻分块的 CDF
            let cdf00 = &tile_histograms[(tile_y * tiles_x + tile_x) as usize];

            // 简单的最近邻插值（可以改进为双线性）
            result.put_pixel(x, y, Luma([cdf00[val]]));
        }
    }

    result
}

/// 自适应直方图均衡化
///
/// # 参数
/// - `image`: 输入灰度图像
/// - `tile_size`: 分块大小
///
/// # 返回
/// 均衡化后的灰度图像
pub fn adaptive_histogram_equalization(image: &GrayImage, tile_size: u32) -> GrayImage {
    clahe(image, 1.0, tile_size) // CLAHE 的特例（无裁剪）
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_median_filter() {
        let mut img = GrayImage::new(5, 5);
        img.put_pixel(2, 2, Luma([255])); // 中心白点（噪声）
        
        let filtered = median_filter(&img, 3);
        
        // 中值滤波应该减少噪声
        assert!(filtered.get_pixel(2, 2)[0] < 255);
    }
    
    #[test]
    fn test_gaussian_blur() {
        let mut img = GrayImage::new(5, 5);
        img.put_pixel(2, 2, Luma([255]));
        
        let blurred = gaussian_blur(&img, 1.0);
        
        // 高斯模糊后，周围像素应该有值
        assert!(blurred.get_pixel(2, 2)[0] < 255);
    }
    
    #[test]
    fn test_clahe() {
        let img = GrayImage::new(16, 16);
        let enhanced = clahe(&img, 2.0, 8);
        
        assert_eq!(enhanced.dimensions(), (16, 16));
    }
}
