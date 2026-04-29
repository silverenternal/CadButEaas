//! 骨架化算法
//!
//! 实现 Zhang-Suen 细化算法，保证 1px 宽度骨架，保留拓扑结构，
//! 并支持亚像素精度边缘定位。

use image::{GrayImage, Luma};

/// 骨架化算法类型
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SkeletonAlgorithm {
    /// Zhang-Suen 细化算法（默认，适合工程图纸）
    #[default]
    ZhangSuen,
    /// Guo-Hall 细化算法（速度更快，可能有更多毛刺）
    GuoHall,
}

/// 骨架化配置
#[derive(Debug, Clone)]
pub struct SkeletonConfig {
    /// 使用的算法
    pub algorithm: SkeletonAlgorithm,
    /// 是否启用去毛刺后处理
    pub enable_de_spur: bool,
    /// 毛刺最大长度（像素）
    pub max_spur_length: usize,
    /// 是否启用亚像素精度
    pub enable_subpixel: bool,
}

impl Default for SkeletonConfig {
    fn default() -> Self {
        Self {
            algorithm: SkeletonAlgorithm::ZhangSuen,
            enable_de_spur: true,
            max_spur_length: 3,
            enable_subpixel: true,
        }
    }
}

// ==================== Zhang-Suen 算法 ====================

/// Zhang-Suen 细化算法（完整实现）
///
/// 特点：
/// - 保证 1px 宽度骨架
/// - 保留 T 型交叉、X 型交叉等拓扑结构
/// - 迭代直到收敛
#[tracing::instrument(name = "zhang_suen", skip(image), level = "debug")]
pub fn skeletonize_zhang_suen(image: &GrayImage) -> GrayImage {
    let (width, height) = image.dimensions();
    let mut result = image.clone();
    let mut changed = true;

    // 预处理：确保前景是黑色（0），背景是白色（255）
    // 如果图像平均颜色较深，需要反转
    let mut foreground_count = 0;
    for pixel in result.pixels() {
        if pixel[0] < 128 {
            foreground_count += 1;
        }
    }

    // 如果前景是白色，反转图像
    let should_invert = foreground_count > (width * height) / 2;
    if should_invert {
        for pixel in result.pixels_mut() {
            pixel[0] = 255 - pixel[0];
        }
    }

    let mut iteration = 0;
    while changed && iteration < 100 {
        changed = false;
        iteration += 1;

        // 子迭代 1
        let mut to_delete = Vec::new();
        for y in 1..(height - 1) {
            for x in 1..(width - 1) {
                if is_foreground(&result, x, y) {
                    let neighbors = get_neighbor_mask(&result, x, y);
                    let p_count = count_foreground_neighbors(neighbors);
                    let transitions = count_transitions(neighbors);

                    // Zhang-Suen 条件 1
                    if (2..=6).contains(&p_count)
                        && transitions == 1
                        && !is_foreground_from_mask(neighbors, 0)
                        && !is_foreground_from_mask(neighbors, 2)
                        && !is_foreground_from_mask(neighbors, 4)
                        && !is_foreground_from_mask(neighbors, 6)
                    {
                        to_delete.push((x, y));
                    }
                }
            }
        }

        for &(x, y) in &to_delete {
            result.put_pixel(x, y, Luma([255]));
            changed = true;
        }

        // 子迭代 2
        let mut to_delete = Vec::new();
        for y in 1..(height - 1) {
            for x in 1..(width - 1) {
                if is_foreground(&result, x, y) {
                    let neighbors = get_neighbor_mask(&result, x, y);
                    let p_count = count_foreground_neighbors(neighbors);
                    let transitions = count_transitions(neighbors);

                    // Zhang-Suen 条件 2
                    if (2..=6).contains(&p_count)
                        && transitions == 1
                        && !is_foreground_from_mask(neighbors, 0)
                        && !is_foreground_from_mask(neighbors, 4)
                        && !is_foreground_from_mask(neighbors, 6)
                    {
                        to_delete.push((x, y));
                    }
                }
            }
        }

        for &(x, y) in &to_delete {
            result.put_pixel(x, y, Luma([255]));
            changed = true;
        }
    }

    tracing::debug!(iterations = iteration, "Zhang-Suen 骨架化完成");

    // 如果之前反转了，现在反转回来
    if should_invert {
        for pixel in result.pixels_mut() {
            pixel[0] = 255 - pixel[0];
        }
    }

    result
}

// ==================== Guo-Hall 算法 ====================

/// Guo-Hall 细化算法（快速版本）
#[tracing::instrument(name = "guo_hall", skip(image), level = "debug")]
pub fn skeletonize_guo_hall(image: &GrayImage) -> GrayImage {
    let (width, height) = image.dimensions();
    let mut result = image.clone();
    let mut changed = true;

    let mut iteration = 0;
    while changed && iteration < 50 {
        changed = false;
        iteration += 1;

        // Guo-Hall 子迭代 1
        let mut to_delete = Vec::new();
        for y in 1..(height - 1) {
            for x in 1..(width - 1) {
                if is_foreground(&result, x, y) {
                    let neighbors = get_neighbor_mask(&result, x, y);
                    let p_count = count_foreground_neighbors(neighbors);
                    let transitions = count_transitions(neighbors);

                    let c1 = (2..=6).contains(&p_count);
                    let c2 = transitions == 1;
                    let c3 = !is_foreground_from_mask(neighbors, 0)
                        || !is_foreground_from_mask(neighbors, 2)
                        || !is_foreground_from_mask(neighbors, 6);
                    let c4 = !is_foreground_from_mask(neighbors, 0)
                        || !is_foreground_from_mask(neighbors, 4)
                        || !is_foreground_from_mask(neighbors, 6);

                    if c1 && c2 && c3 && c4 {
                        to_delete.push((x, y));
                    }
                }
            }
        }

        for &(x, y) in &to_delete {
            result.put_pixel(x, y, Luma([255]));
            changed = true;
        }

        // Guo-Hall 子迭代 2
        let mut to_delete = Vec::new();
        for y in 1..(height - 1) {
            for x in 1..(width - 1) {
                if is_foreground(&result, x, y) {
                    let neighbors = get_neighbor_mask(&result, x, y);
                    let p_count = count_foreground_neighbors(neighbors);
                    let transitions = count_transitions(neighbors);

                    let c1 = (2..=6).contains(&p_count);
                    let c2 = transitions == 1;
                    let c3 = !is_foreground_from_mask(neighbors, 0)
                        || !is_foreground_from_mask(neighbors, 2)
                        || !is_foreground_from_mask(neighbors, 4);
                    let c4 = !is_foreground_from_mask(neighbors, 2)
                        || !is_foreground_from_mask(neighbors, 4)
                        || !is_foreground_from_mask(neighbors, 6);

                    if c1 && c2 && c3 && c4 {
                        to_delete.push((x, y));
                    }
                }
            }
        }

        for &(x, y) in &to_delete {
            result.put_pixel(x, y, Luma([255]));
            changed = true;
        }
    }

    tracing::debug!(iterations = iteration, "Guo-Hall 骨架化完成");

    result
}

// ==================== 辅助函数 ====================

/// 获取 8 邻域掩码（顺时针排列，从上方开始）
///
/// ```text
/// 7 0 1
/// 6 P 2
/// 5 4 3
/// ```
#[inline(always)]
fn get_neighbor_mask(image: &GrayImage, x: u32, y: u32) -> u8 {
    let mut mask = 0u8;

    // P0: 上
    mask |= if is_foreground(image, x, y - 1) {
        1 << 0
    } else {
        0
    };
    // P1: 右上
    mask |= if is_foreground(image, x + 1, y - 1) {
        1 << 1
    } else {
        0
    };
    // P2: 右
    mask |= if is_foreground(image, x + 1, y) {
        1 << 2
    } else {
        0
    };
    // P3: 右下
    mask |= if is_foreground(image, x + 1, y + 1) {
        1 << 3
    } else {
        0
    };
    // P4: 下
    mask |= if is_foreground(image, x, y + 1) {
        1 << 4
    } else {
        0
    };
    // P5: 左下
    mask |= if is_foreground(image, x - 1, y + 1) {
        1 << 5
    } else {
        0
    };
    // P6: 左
    mask |= if is_foreground(image, x - 1, y) {
        1 << 6
    } else {
        0
    };
    // P7: 左上
    mask |= if is_foreground(image, x - 1, y - 1) {
        1 << 7
    } else {
        0
    };

    mask
}

/// 检查邻域点是否是前景
#[inline(always)]
fn is_foreground_from_mask(mask: u8, idx: u8) -> bool {
    (mask & (1 << idx)) != 0
}

/// 计数前景邻域数量
#[inline(always)]
fn count_foreground_neighbors(mask: u8) -> u8 {
    mask.count_ones() as u8
}

/// 计数 0->1 模式转换次数
#[inline(always)]
fn count_transitions(mask: u8) -> u8 {
    let mut count = 0;
    let mut prev = (mask >> 7) & 1; // P7 作为起始

    for i in 0..8 {
        let curr = (mask >> i) & 1;
        if prev == 0 && curr == 1 {
            count += 1;
        }
        prev = curr;
    }

    count
}

/// 检查是否是前景像素（黑色 = 0）
#[inline(always)]
fn is_foreground(image: &GrayImage, x: u32, y: u32) -> bool {
    image.get_pixel(x, y)[0] < 128
}

// ==================== 去毛刺后处理 ====================

/// 移除骨架上的短毛刺
#[tracing::instrument(name = "de_spur", skip(image), level = "debug")]
pub fn remove_spurs(image: &GrayImage, max_length: usize) -> GrayImage {
    let mut result = image.clone();
    let (width, height) = image.dimensions();

    let mut removed = 0;
    let mut changed = true;

    while changed {
        changed = false;
        let mut endpoints = Vec::new();

        // 找到所有端点（度数为 1 的像素）
        for y in 1..(height - 1) {
            for x in 1..(width - 1) {
                if is_foreground(&result, x, y) {
                    let neighbors = get_neighbor_mask(&result, x, y);
                    let count = count_foreground_neighbors(neighbors);
                    if count == 1 {
                        endpoints.push((x, y));
                    }
                }
            }
        }

        // 从每个端点沿着分支行走，标记短分支
        for &(x, y) in &endpoints {
            let branch = trace_branch(&result, x, y, max_length);
            if branch.len() <= max_length {
                // 移除短分支
                for &(bx, by) in &branch {
                    result.put_pixel(bx, by, Luma([255]));
                    removed += 1;
                }
                changed = true;
            }
        }
    }

    if removed > 0 {
        tracing::debug!(spurs_removed = removed, "毛刺移除完成");
    }

    result
}

/// 从端点追踪分支，最多追踪 max_length 步
fn trace_branch(
    image: &GrayImage,
    start_x: u32,
    start_y: u32,
    max_length: usize,
) -> Vec<(u32, u32)> {
    let mut branch = Vec::new();
    let mut x = start_x;
    let mut y = start_y;
    let mut prev_x = u32::MAX;
    let mut prev_y = u32::MAX;

    while branch.len() <= max_length {
        branch.push((x, y));

        let neighbors = get_neighbor_mask(image, x, y);
        let mut next_points = Vec::new();

        // 找到所有下一个可能的点（排除来路）
        for (i, (dx, dy)) in [
            (0i32, -1i32), // 上 (P0)
            (1, -1),       // 右上 (P1)
            (1, 0),        // 右 (P2)
            (1, 1),        // 右下 (P3)
            (0, 1),        // 下 (P4)
            (-1, 1),       // 左下 (P5)
            (-1, 0),       // 左 (P6)
            (-1, -1),      // 左上 (P7)
        ]
        .iter()
        .enumerate()
        {
            if is_foreground_from_mask(neighbors, i as u8) {
                let nx = (x as i32 + dx) as u32;
                let ny = (y as i32 + dy) as u32;
                if nx != prev_x || ny != prev_y {
                    next_points.push((nx, ny));
                }
            }
        }

        if next_points.is_empty() {
            break;
        }

        // 如果到达交叉点（>1 个出口），停止
        if next_points.len() > 1 {
            break;
        }

        // 继续前进
        prev_x = x;
        prev_y = y;
        x = next_points[0].0;
        y = next_points[0].1;
    }

    branch
}

// ==================== 亚像素精度 ====================

/// 亚像素坐标点
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SubpixelPoint {
    pub x: f64,
    pub y: f64,
}

impl SubpixelPoint {
    /// 创建新点
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    /// 计算两点距离
    pub fn distance(&self, other: &Self) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        (dx * dx + dy * dy).sqrt()
    }
}

impl From<(f64, f64)> for SubpixelPoint {
    fn from((x, y): (f64, f64)) -> Self {
        Self { x, y }
    }
}

/// 亚像素精度边缘定位
///
/// 使用灰度矩方法计算骨架点的精确重心位置，达到亚像素精度。
/// 在 3x3 窗口内计算灰度重心，修正骨架位置。
#[tracing::instrument(name = "subpixel_refine", skip(image, skeleton), level = "debug")]
pub fn refine_subpixel(image: &GrayImage, skeleton: &GrayImage) -> Vec<SubpixelPoint> {
    let (width, height) = image.dimensions();
    let mut points = Vec::new();

    for y in 1..(height - 1) {
        for x in 1..(width - 1) {
            if is_foreground(skeleton, x, y) {
                // 在原始灰度图中计算 3x3 窗口的灰度重心
                let mut sum_x = 0.0;
                let mut sum_y = 0.0;
                let mut sum_weight = 0.0;

                for dy in -1isize..=1isize {
                    for dx in -1isize..=1isize {
                        let nx = (x as isize + dx) as u32;
                        let ny = (y as isize + dy) as u32;
                        // 权重：越暗（越接近边缘）权重越高
                        let weight = (255 - image.get_pixel(nx, ny)[0]) as f64 / 255.0;
                        let weight = weight * weight; // 强调边缘

                        sum_x += dx as f64 * weight;
                        sum_y += dy as f64 * weight;
                        sum_weight += weight;
                    }
                }

                if sum_weight > 1e-6 {
                    let sub_x = sum_x / sum_weight;
                    let sub_y = sum_y / sum_weight;

                    points.push(SubpixelPoint {
                        x: x as f64 + sub_x,
                        y: y as f64 + sub_y,
                    });
                } else {
                    // 权重太小，使用整数坐标
                    points.push(SubpixelPoint {
                        x: x as f64,
                        y: y as f64,
                    });
                }
            }
        }
    }

    tracing::debug!(points_count = points.len(), "亚像素点提取完成");

    points
}

// ==================== 统一入口 ====================

/// 骨架化统一入口
#[tracing::instrument(name = "skeletonize", skip(image), level = "debug")]
pub fn skeletonize(image: &GrayImage, config: &SkeletonConfig) -> GrayImage {
    let start = std::time::Instant::now();

    let mut skeleton = match config.algorithm {
        SkeletonAlgorithm::ZhangSuen => skeletonize_zhang_suen(image),
        SkeletonAlgorithm::GuoHall => skeletonize_guo_hall(image),
    };

    if config.enable_de_spur {
        skeleton = remove_spurs(&skeleton, config.max_spur_length);
    }

    let duration = start.elapsed();
    tracing::info!(
        target: "vectorize::performance",
        duration_ms = duration.as_millis(),
        image_size = image.width() * image.height(),
        algorithm = ?config.algorithm,
        "骨架化完成"
    );

    skeleton
}

/// 骨架化 + 亚像素定位完整流程
pub fn skeletonize_with_subpixel(
    image: &GrayImage,
    config: &SkeletonConfig,
) -> (GrayImage, Vec<SubpixelPoint>) {
    let skeleton = skeletonize(image, config);
    let subpixel_points = if config.enable_subpixel {
        refine_subpixel(image, &skeleton)
    } else {
        Vec::new()
    };
    (skeleton, subpixel_points)
}

// ==================== 测试 ====================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zhang_suen_horizontal_line() {
        // 水平直线 - 初始化背景为白色
        let mut img = GrayImage::from_pixel(10, 5, Luma([255]));
        for x in 0..10 {
            img.put_pixel(x, 2, Luma([0]));
        }

        let result = skeletonize_zhang_suen(&img);

        // 应该保持为 1px 直线
        for x in 1..9 {
            assert!(
                is_foreground(&result, x, 2),
                "Line at x={} should remain",
                x
            );
        }
    }

    #[test]
    fn test_zhang_suen_vertical_line() {
        // 垂直直线 - 初始化背景为白色
        let mut img = GrayImage::from_pixel(5, 10, Luma([255]));
        for y in 0..10 {
            img.put_pixel(2, y, Luma([0]));
        }

        let result = skeletonize_zhang_suen(&img);

        // 应该保持为 1px 直线
        for y in 1..9 {
            assert!(
                is_foreground(&result, 2, y),
                "Line at y={} should remain",
                y
            );
        }
    }

    #[test]
    fn test_t_junction_preservation() {
        // T 型交叉 - 初始化背景为白色
        let mut img = GrayImage::from_pixel(7, 7, Luma([255]));
        // 水平线
        for x in 1..6 {
            img.put_pixel(x, 3, Luma([0]));
        }
        // 垂直线
        for y in 3..6 {
            img.put_pixel(3, y, Luma([0]));
        }

        let result = skeletonize_zhang_suen(&img);

        // 交叉点应该保留
        assert!(
            is_foreground(&result, 3, 3),
            "T-junction should be preserved"
        );
    }

    #[test]
    fn test_spur_removal() {
        // 创建带毛刺的图像 - 初始化背景为白色
        let mut img = GrayImage::from_pixel(10, 10, Luma([255]));
        // 主线
        for x in 1..9 {
            img.put_pixel(x, 5, Luma([0]));
        }
        // 短毛刺 (2px)
        img.put_pixel(5, 4, Luma([0]));
        img.put_pixel(5, 3, Luma([0]));

        let result = remove_spurs(&img, 2);

        // 毛刺应该被移除
        assert!(!is_foreground(&result, 5, 3), "Spur tip should be removed");
        assert!(!is_foreground(&result, 5, 4), "Spur base should be removed");
        // 主线应该保留
        assert!(is_foreground(&result, 4, 5), "Main line should remain");
    }

    #[test]
    fn test_subpixel_refine() {
        let mut img = GrayImage::from_pixel(5, 5, Luma([255]));
        // 中心像素周围有灰度梯度
        img.put_pixel(2, 2, Luma([0]));
        img.put_pixel(1, 2, Luma([64]));
        img.put_pixel(3, 2, Luma([64]));
        img.put_pixel(2, 1, Luma([64]));
        img.put_pixel(2, 3, Luma([64]));

        // 骨架 - 初始化背景为白色
        let mut skeleton = GrayImage::from_pixel(5, 5, Luma([255]));
        skeleton.put_pixel(2, 2, Luma([0]));

        let points = refine_subpixel(&img, &skeleton);

        // 应该提取到一个亚像素点，位置应该接近 (2.0, 2.0)
        assert_eq!(points.len(), 1);
        assert!((points[0].x - 2.0).abs() < 0.5);
        assert!((points[0].y - 2.0).abs() < 0.5);
    }

    #[test]
    fn test_count_transitions() {
        // 测试 0->1 转换计数
        // 模式：只有 P2 是前景，其他是背景
        // 应该只有 1 次转换
        let mask = 1 << 2; // 0b00000100
        assert_eq!(count_transitions(mask), 1);

        // 模式：P0 和 P4 是前景
        // P0=1, P1-P3=0, P4=1, P5-P7=0
        // 序列: 0 0 0 0 1 0 0 1 -> 注意：从 P7 开始循环
        // 实际上应该是 2 次转换
        let mask = (1 << 0) | (1 << 4); // P0 和 P4
        assert_eq!(count_transitions(mask), 2);
    }

    #[test]
    fn test_neighbor_mask() {
        let mut img = GrayImage::from_pixel(3, 3, Luma([255]));
        // 设置特定邻域点
        img.put_pixel(1, 0, Luma([0])); // P0 (上)
        img.put_pixel(2, 1, Luma([0])); // P2 (右)
        img.put_pixel(1, 1, Luma([0])); // 中心点

        let mask = get_neighbor_mask(&img, 1, 1);

        assert!(is_foreground_from_mask(mask, 0), "P0 should be foreground");
        assert!(is_foreground_from_mask(mask, 2), "P2 should be foreground");
        assert!(!is_foreground_from_mask(mask, 4), "P4 should be background");
    }
}
