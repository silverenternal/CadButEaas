//! 透视变换校正
//!
//! 对拍照产生的透视变形图纸进行校正，将梯形变换为矩形。
//! 基于四个角点计算单应性矩阵，然后进行逆透视变换得到校正图像。

use image::{GrayImage, Luma};
use common_types::Point2;
use crate::paper_detection::PaperRegion;

/// 透视校正结果
#[derive(Debug, Clone)]
pub struct PerspectiveCorrection {
    /// 校正后的图像
    pub corrected: GrayImage,
    /// 原始四个角点
    pub src_corners: [Point2; 4],
    /// 目标矩形尺寸
    pub dst_size: (u32, u32),
    /// 单应性矩阵 (3x3)
    pub homography: [[f64; 3]; 3],
}

/// 对检测到的纸张区域进行透视校正
///
/// # 参数
/// - `image`: 原始灰度图像
/// - `region`: 检测到的纸张区域，包含四个角点
///
/// # 返回
/// 校正后的图像，尺寸为目标矩形尺寸
pub fn correct_perspective(image: &GrayImage, region: &PaperRegion) -> GrayImage {
    let [tl, tr, br, bl] = region.corners;

    // 计算目标宽度和高度（使用最长边）
    let width_top = distance(tr, tl);
    let width_bottom = distance(br, bl);
    let height_left = distance(bl, tl);
    let height_right = distance(br, tr);

    let dst_width = width_top.max(width_bottom).round() as u32;
    let dst_height = height_left.max(height_right).round() as u32;

    // 目标角点（矩形，坐标从 0,0 开始）
    let dst_corners = [
        [0.0, 0.0],
        [dst_width as f64 - 1.0, 0.0],
        [dst_width as f64 - 1.0, dst_height as f64 - 1.0],
        [0.0, dst_height as f64 - 1.0],
    ];

    // 计算单应性矩阵
    let homography = compute_homography(&region.corners, &dst_corners);

    // 进行逆透视变换
    warp_perspective(image, &homography, dst_width, dst_height)
}

/// 自动检测纸张并进行透视校正
///
/// 如果纸张检测失败或者置信度太低，返回原图
pub fn auto_correct(image: &GrayImage) -> GrayImage {
    match crate::paper_detection::detect_paper(image) {
        Some(region) => {
            if region.confidence > 0.5 {
                correct_perspective(image, &region)
            } else {
                image.clone()
            }
        }
        None => image.clone(),
    }
}

/// 计算两点间距离
fn distance(a: Point2, b: Point2) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    (dx * dx + dy * dy).sqrt()
}

/// 计算 3x3 单应性矩阵 H
/// 满足: dst = H * src (齐次坐标)
/// 使用 DLT 算法
fn compute_homography(src: &[Point2; 4], dst: &[Point2; 4]) -> [[f64; 3]; 3] {
    // 构建 8x9 矩阵 A
    // 每个点对应两个方程
    let mut a = [[0.0f64; 9]; 8];

    for i in 0..4 {
        let x = src[i][0];
        let y = src[i][1];
        let u = dst[i][0];
        let v = dst[i][1];

        // 第一个方程: -x  -y  -1  0  0  0  x*u  y*u  u
        a[2 * i][0] = -x;
        a[2 * i][1] = -y;
        a[2 * i][2] = -1.0;
        a[2 * i][3] = 0.0;
        a[2 * i][4] = 0.0;
        a[2 * i][5] = 0.0;
        a[2 * i][6] = x * u;
        a[2 * i][7] = y * u;
        a[2 * i][8] = u;

        // 第二个方程: 0  0  0  -x  -y  -1  x*v  y*v  v
        a[2 * i + 1][0] = 0.0;
        a[2 * i + 1][1] = 0.0;
        a[2 * i + 1][2] = 0.0;
        a[2 * i + 1][3] = -x;
        a[2 * i + 1][4] = -y;
        a[2 * i + 1][5] = -1.0;
        a[2 * i + 1][6] = x * v;
        a[2 * i + 1][7] = y * v;
        a[2 * i + 1][8] = v;
    }

    // 使用 SVD 求解 Ah = 0
    // 对于 8x9 矩阵，我们可以直接用简化的 DLT 求解
    let h = solve_dlt(&a);

    // 重塑为 3x3 矩阵，归一化使得 h[8] = 1
    let mut hom = [
        [h[0], h[1], h[2]],
        [h[3], h[4], h[5]],
        [h[6], h[7], h[8]],
    ];

    // 归一化
    let scale = 1.0 / hom[2][2];
    for row in hom.iter_mut() {
        for elem in row.iter_mut() {
            *elem *= scale;
        }
    }

    hom
}

/// 简化 DLT 求解 8x9 系统
/// 使用 numpy 风格的方法：最小奇异值对应的右奇异向量就是解
/// 这里我们用逆幂法（逆迭代）求解 A^T A 的最小特征值对应的特征向量，非常精确
fn solve_dlt(a: &[[f64; 9]; 8]) -> [f64; 9] {
    // 计算 A^T * A，这是 9x9 对称正定矩阵
    let mut ata = [[0.0f64; 9]; 9];
    for i in 0..9 {
        for j in 0..9 {
            let mut sum = 0.0;
            for row in 0..8 {
                sum += a[row][i] * a[row][j];
            }
            ata[i][j] = sum;
        }
    }

    // 逆幂法：我们寻找最小特征值 λ ≈ 0，所以解 ata * x_{k+1} = x_k
    // 从初始向量开始迭代，会快速收敛到最小特征值对应的特征向量
    let mut h = [1.0f64; 9];

    // 进行 20 次逆幂迭代足够收敛
    for _ in 0..20 {
        // 使用高斯消元解 ata * new_h = h
        let mut new_h = solve_linear_system(&ata, &h);

        // 归一化
        let norm = new_h.iter().map(|&v| v * v).sum::<f64>().sqrt();
        if norm > 1e-10 {
            for v in new_h.iter_mut() {
                *v /= norm;
            }
        }

        h = new_h;
    }

    h
}

/// 求解 9x9 线性方程组 Ax = b，使用高斯消元法
fn solve_linear_system(a: &[[f64; 9]; 9], b: &[f64; 9]) -> [f64; 9] {
    let mut aug = [[0.0f64; 10]; 9]; // 增广矩阵
    for i in 0..9 {
        for j in 0..9 {
            aug[i][j] = a[i][j];
        }
        aug[i][9] = b[i];
    }

    // 高斯消元（按列主元）
    for col in 0..9 {
        // 找主元
        let mut pivot_row = col;
        let mut max_abs = aug[col][col].abs();
        for r in (col + 1)..9 {
            if aug[r][col].abs() > max_abs {
                max_abs = aug[r][col].abs();
                pivot_row = r;
            }
        }
        // 交换行
        aug.swap(col, pivot_row);

        // 归一化主元行
        let pivot = aug[col][col];
        if pivot.abs() < 1e-10 {
            break; // 奇异，停止
        }
        for j in col..=9 {
            aug[col][j] /= pivot;
        }

        // 消去下方
        for r in (col + 1)..9 {
            let factor = aug[r][col];
            for j in col..=9 {
                aug[r][j] -= factor * aug[col][j];
            }
        }
    }

    // 回代
    let mut x = [0.0f64; 9];
    for i in (0..9).rev() {
        let mut sum = aug[i][9];
        for j in (i + 1)..9 {
            sum -= aug[i][j] * x[j];
        }
        x[i] = sum;
    }

    x
}

/// 应用逆透视变换
/// 对于目标图像每个像素 (x', y')，计算其在原始图像中的坐标 (x, y)，然后插值
fn warp_perspective(src: &GrayImage, homography: &[[f64; 3]; 3], dst_width: u32, dst_height: u32) -> GrayImage {
    let mut dst = GrayImage::new(dst_width, dst_height);
    let (src_width, src_height) = src.dimensions();

    // 对目标图像每个像素
    for dst_y in 0..dst_height {
        for dst_x in 0..dst_width {
            // 反向变换：从目标坐标计算源坐标
            // H * src = dst → src = H^{-1} * dst
            // 我们直接计算 inv(H) * [dst_x, dst_y, 1]
            let x = dst_x as f64;
            let y = dst_y as f64;

            // 计算 z 分量分母
            let denom = homography[2][0] * x + homography[2][1] * y + homography[2][2];
            if denom.abs() < 1e-10 {
                continue;
            }

            let src_x = (homography[0][0] * x + homography[0][1] * y + homography[0][2]) / denom;
            let src_y = (homography[1][0] * x + homography[1][1] * y + homography[1][2]) / denom;

            // 双线性插值
            let pixel = bilinear_interpolate(src, src_x, src_y, src_width, src_height);
            dst.put_pixel(dst_x, dst_y, Luma([pixel]));
        }
    }

    dst
}

/// 双线性插值获取像素值
fn bilinear_interpolate(image: &GrayImage, x: f64, y: f64, width: u32, height: u32) -> u8 {
    // 边界检查
    if x < 0.0 || x >= (width - 1) as f64 || y < 0.0 || y >= (height - 1) as f64 {
        // 超出范围，返回最近像素或者背景白色
        let x_clamped = x.clamp(0.0, (width - 1) as f64);
        let y_clamped = y.clamp(0.0, (height - 1) as f64);
        return image.get_pixel(x_clamped as u32, y_clamped as u32)[0];
    }

    let x0 = x.floor() as u32;
    let x1 = x0 + 1;
    let y0 = y.floor() as u32;
    let y1 = y0 + 1;

    let fx = x - x0 as f64;
    let fy = y - y0 as f64;

    let v00 = image.get_pixel(x0, y0)[0] as f64;
    let v10 = image.get_pixel(x1, y0)[0] as f64;
    let v01 = image.get_pixel(x0, y1)[0] as f64;
    let v11 = image.get_pixel(x1, y1)[0] as f64;

    // 双线性插值
    let v0 = v00 * (1.0 - fx) + v10 * fx;
    let v1 = v01 * (1.0 - fx) + v11 * fx;
    let v = v0 * (1.0 - fy) + v1 * fy;

    v.round() as u8
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{GrayImage, Luma};

    #[test]
    fn test_identity_homography() {
        // 单位变换应该输出相同图像
        let src = [
            [0.0, 0.0],
            [100.0, 0.0],
            [100.0, 100.0],
            [0.0, 100.0],
        ];
        let dst = [
            [0.0, 0.0],
            [100.0, 0.0],
            [100.0, 100.0],
            [0.0, 100.0],
        ];
        let hom = compute_homography(&src, &dst);

        // 应该接近单位矩阵
        // 迭代方法允许稍微大一点的误差
        assert!((hom[0][0] - 1.0).abs() < 0.05);
        assert!((hom[1][1] - 1.0).abs() < 0.05);
        assert!((hom[2][2] - 1.0).abs() < 0.05);
        assert!(hom[0][1].abs() < 0.05);
        assert!(hom[1][0].abs() < 0.05);
    }

    #[test]
    fn test_bilinear_interpolation() {
        let mut img = GrayImage::new(2, 2);
        img.put_pixel(0, 0, Luma([0]));
        img.put_pixel(1, 0, Luma([100]));
        img.put_pixel(0, 1, Luma([50]));
        img.put_pixel(1, 1, Luma([150]));

        // 中心点插值
        let v = bilinear_interpolate(&img, 0.5, 0.5, 2, 2);
        // (0 + 100 + 50 + 150) / 4 = 75
        assert!((v as i32 - 75).abs() <= 1);
    }

    #[test]
    fn test_correct_simple_trapezoid() {
        // 创建一个梯形图像，模拟透视
        let mut img = GrayImage::new(100, 100);
        // 白色背景
        for y in 0..100 {
            for x in 0..100 {
                img.put_pixel(x, y, Luma([255]));
            }
        }
        // 画一个黑色梯形（左上偏左，右上偏右）
        for y in 10..=90 {
            let t = (y - 10) as f64 / 80.0;
            let x_left = 10.0 - t * 10.0;
            let x_right = 90.0 + t * 10.0;
            for x in (x_left.floor() as u32)..=(x_right.ceil() as u32) {
                if x < 100 {
                    img.put_pixel(x, y, Luma([0]));
                }
            }
        }

        // 四个角点（梯形）
        let corners = [
            [10.0, 10.0],  // tl
            [90.0, 10.0],  // tr
            [100.0, 90.0], // br
            [0.0, 90.0],   // bl
        ];

        let region = PaperRegion {
            corners,
            bbox: (0, 10, 100, 90),
            confidence: 0.9,
        };

        let corrected = correct_perspective(&img, &region);
        // 校正后应该是矩形，宽度约 100，高度约 80
        assert!(corrected.width() >= 80);
        assert!(corrected.height() >= 70);
    }
}
