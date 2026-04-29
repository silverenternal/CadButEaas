//! 矢量化算法
//!
//! 提供纯 Rust 实现，支持可选的 OpenCV 加速

#![allow(clippy::needless_range_loop)]

// OpenCV 配置（仅在启用 feature 时）
#[cfg(feature = "opencv")]
pub mod opencv_config;

// 预处理算法
pub mod preprocessing;
pub use preprocessing::*;

// 阈值算法
pub mod threshold;
pub use threshold::*;

// 骨架化算法
pub mod skeleton;
pub use skeleton::*;

// 轨迹追踪
pub mod tracing;
pub use tracing::*;

// 基元拟合引擎
pub mod primitive_fitting;
pub use primitive_fitting::*;

// 线型识别
pub mod line_type;
pub use line_type::*;

// 圆弧拟合
pub mod arc_fitting;
pub use arc_fitting::*;

// 断点连接
pub mod gap_filling;
pub use gap_filling::*;

// 质量评估
pub mod quality;
pub use quality::*;

// NURBS 曲率自适应离散化（P1-2）
pub mod nurbs_adaptive;
pub use nurbs_adaptive::*;

// 文字标注分离
pub mod text_blob;
pub use text_blob::*;

// OCR 文本识别与空间关联
pub mod ocr;
pub use ocr::*;

// VLM 尺寸标注解析
pub mod vlm;
pub use vlm::*;

// CAD 符号库与模板匹配
pub mod symbols;
pub use symbols::*;

// 拓扑缝合算法
pub mod topology_stitch;
pub use topology_stitch::*;

// Halfedge 网格转换器
pub mod halfedge_convert;
pub use halfedge_convert::*;

// 矢量化质量评估指标体系（几何/拓扑/语义）
pub mod quality_eval;
pub use quality_eval::*;

// 自适应参数调整（基于图像质量）
pub mod adaptive_params;
pub use adaptive_params::*;

// 纸张检测与自动裁剪
pub mod paper_detection;
pub use paper_detection::*;

// 透视变换校正
pub mod perspective_correction;
pub use perspective_correction::*;

// 建筑规则几何校正（正交性/平行性）
pub mod architectural_rules;
pub use architectural_rules::*;

use common_types::{Point2, Polyline};
use image::GrayImage;
use rayon::prelude::*;

#[cfg(feature = "opencv")]
mod opencv_impl {
    use super::*;
    use opencv::{
        core::{Mat, MatTraitConst, MatTraitManual, Point, Scalar, Size, Vector},
        imgproc::{
            self, approx_poly_dp, canny, threshold as cv_threshold, THRESH_BINARY, THRESH_OTSU,
        },
        prelude::{MatExprTraitConst, MatTraitConstManual},
    };

    /// 快速将 GrayImage 转换为 OpenCV Mat（批量内存拷贝）
    fn gray_image_to_mat(image: &GrayImage) -> Result<Mat, String> {
        let (_width, height) = image.dimensions();
        let raw_data = image.as_raw();

        // 使用 Mat::from_slice 直接包装内存，避免拷贝
        let mat = Mat::from_slice(raw_data)
            .map_err(|e| format!("Mat from_slice 失败：{}", e))?
            .try_clone()
            .map_err(|e| format!("clone Mat 失败：{}", e))?;

        // 重塑为正确尺寸（Mat::from_slice 默认是 1D 数组）
        let reshaped = mat
            .reshape(1, height as i32) // 1 通道，height 行
            .map_err(|e| format!("reshape 失败：{}", e))?
            .try_clone()
            .map_err(|e| format!("clone reshaped Mat 失败：{}", e))?;

        Ok(reshaped)
    }

    /// 快速将 OpenCV Mat 转换回 GrayImage
    fn mat_to_gray_image(mat: &Mat) -> Result<GrayImage, String> {
        let height = mat.rows();
        let width = mat.cols();

        // 直接读取连续内存
        let data = mat
            .data_typed::<u8>()
            .map_err(|e| format!("data_typed 失败：{}", e))?;

        let mut image = GrayImage::new(width as u32, height as u32);
        let raw = image.as_mut();
        raw.copy_from_slice(data);

        Ok(image)
    }

    /// 快速将 GrayImage 转换为二值 Mat（用于轮廓查找等需要反转的场景）
    fn gray_image_to_binary_mat(image: &GrayImage, invert: bool) -> Result<Mat, String> {
        let (width, height) = image.dimensions();
        let raw_data = image.as_raw();

        // 创建 Mat 并复制数据
        let mut mat = Mat::new_rows_cols_with_default(
            height as i32,
            width as i32,
            opencv::core::CV_8UC1,
            Scalar::all(0.0),
        )
        .map_err(|e| format!("创建 Mat 失败：{}", e))?;

        // 批量拷贝数据
        let mat_data = mat
            .data_typed_mut::<u8>()
            .map_err(|e| format!("获取 Mat 数据失败：{}", e))?;
        mat_data.copy_from_slice(raw_data);

        // 如果需要反转（0->255, 255->0）
        if invert {
            // 创建一个临时 Mat 用于存储反转结果
            let mut result_mat = Mat::default();
            opencv::core::bitwise_not(&mat, &mut result_mat, &opencv::core::no_array())
                .map_err(|e| format!("反转失败：{}", e))?;
            mat = result_mat;
        }

        Ok(mat)
    }

    /// 使用 OpenCV 进行边缘检测（Canny 算子）
    #[tracing::instrument(name = "opencv_canny", skip(image), level = "debug")]
    pub fn detect_edges_opencv(image: &GrayImage) -> Result<GrayImage, String> {
        let start = std::time::Instant::now();
        let mat = gray_image_to_mat(image)?;

        let mut edges = Mat::default();
        canny(&mat, &mut edges, 50.0, 150.0, 3, false)
            .map_err(|e| format!("Canny 边缘检测失败：{}", e))?;

        let result = mat_to_gray_image(&edges)?;

        let duration = start.elapsed();
        tracing::info!(
            target: "vectorize::performance",
            duration_ms = duration.as_millis(),
            image_size = image.width() * image.height(),
            "OpenCV Canny 完成"
        );

        Ok(result)
    }

    /// 使用 OpenCV 进行自适应阈值二值化
    #[tracing::instrument(name = "opencv_threshold", skip(image), level = "debug")]
    pub fn threshold_opencv(image: &GrayImage, adaptive: bool) -> Result<GrayImage, String> {
        let start = std::time::Instant::now();
        let mat = gray_image_to_mat(image)?;
        let mut result_mat = Mat::default();

        if adaptive {
            let ret = cv_threshold(
                &mat,
                &mut result_mat,
                0.0,
                255.0,
                THRESH_BINARY | THRESH_OTSU,
            )
            .map_err(|e| format!("Otsu 阈值失败：{}", e))?;
            tracing::debug!("Otsu 自动阈值：{}", ret);
        } else {
            cv_threshold(&mat, &mut result_mat, 128.0, 255.0, THRESH_BINARY)
                .map_err(|e| format!("固定阈值失败：{}", e))?;
        }

        let result = mat_to_gray_image(&result_mat)?;

        let duration = start.elapsed();
        tracing::info!(
            target: "vectorize::performance",
            duration_ms = duration.as_millis(),
            image_size = image.width() * image.height(),
            adaptive = adaptive,
            "OpenCV 阈值处理完成"
        );

        Ok(result)
    }

    /// 使用 OpenCV 进行形态学细化（骨架化）
    #[tracing::instrument(name = "opencv_skeleton", skip(image), level = "debug")]
    pub fn skeletonize_opencv(image: &GrayImage) -> Result<GrayImage, String> {
        let start = std::time::Instant::now();
        let (width, height) = image.dimensions();

        // 转换为二值 Mat（反转：黑->白，白->黑）
        let mut mat = gray_image_to_binary_mat(image, true)?;

        let mut skeleton = Mat::zeros(height as i32, width as i32, opencv::core::CV_8UC1)
            .map_err(|e| format!("创建骨架 Mat 失败：{}", e))?
            .to_mat()
            .map_err(|e| format!("转换为 Mat 失败：{}", e))?
            .try_clone()
            .map_err(|e| format!("clone Mat 失败：{}", e))?;
        let mut temp = Mat::default();
        let mut eroded = Mat::default();

        let kernel = imgproc::get_structuring_element(
            imgproc::MORPH_CROSS,
            Size::new(3, 3),
            Point::new(-1, -1),
        )
        .map_err(|e| format!("创建结构元素失败：{}", e))?;

        loop {
            imgproc::erode(
                &mat,
                &mut eroded,
                &kernel,
                Point::new(-1, -1),
                1,
                opencv::core::BORDER_CONSTANT,
                opencv::core::Scalar::all(0.0),
            )
            .map_err(|e| format!("腐蚀失败：{}", e))?;

            imgproc::dilate(
                &eroded,
                &mut temp,
                &kernel,
                Point::new(-1, -1),
                1,
                opencv::core::BORDER_CONSTANT,
                opencv::core::Scalar::all(0.0),
            )
            .map_err(|e| format!("膨胀失败：{}", e))?;

            // 使用单独的临时变量避免借用冲突
            let mut diff = Mat::default();
            opencv::core::subtract(&mat, &temp, &mut diff, &opencv::core::no_array(), -1)
                .map_err(|e| format!("减法失败：{}", e))?;

            // 克隆 skeleton 避免借用冲突
            let skeleton_clone = skeleton
                .try_clone()
                .map_err(|e| format!("clone skeleton 失败：{}", e))?;
            opencv::core::bitwise_or(
                &skeleton_clone,
                &diff,
                &mut skeleton,
                &opencv::core::no_array(),
            )
            .map_err(|e| format!("位或失败：{}", e))?;

            let count =
                opencv::core::count_non_zero(&eroded).map_err(|e| format!("计数失败：{}", e))?;

            mat = eroded.clone();

            if count == 0 {
                break;
            }
        }

        // 转换回 GrayImage（需要反转回来）
        let result = mat_to_gray_image(&skeleton)?;
        let mut final_result = GrayImage::new(width, height);
        for y in 0..height {
            for x in 0..width {
                let val = result.get_pixel(x, y)[0];
                final_result.put_pixel(x, y, Luma([if val > 0 { 0 } else { 255 }]));
            }
        }

        let duration = start.elapsed();
        tracing::info!(
            target: "vectorize::performance",
            duration_ms = duration.as_millis(),
            image_size = width * height,
            "OpenCV 骨架化完成"
        );

        Ok(final_result)
    }

    /// 霍夫变换检测直线
    pub struct HoughLine {
        pub start: Point2,
        pub end: Point2,
        pub confidence: f32,
    }

    #[tracing::instrument(name = "opencv_hough", skip(image), level = "debug")]
    pub fn detect_lines_hough(
        image: &GrayImage,
        min_length: f64,
        threshold: u32,
    ) -> Result<Vec<HoughLine>, String> {
        let start = std::time::Instant::now();
        let mat = gray_image_to_mat(image)?;

        let mut lines = Mat::default();
        imgproc::hough_lines_p(
            &mat,
            &mut lines,
            1.0,
            std::f64::consts::PI / 180.0,
            threshold as f64,
            min_length,
            10.0,
        )
        .map_err(|e| format!("霍夫变换失败：{}", e))?;

        let mut result = Vec::new();
        if lines.total() > 0 {
            // 读取霍夫线数据
            if let Ok(lines_vec) = lines.to_vec_2d::<i32>() {
                for line in lines_vec.iter() {
                    if line.len() >= 4 {
                        result.push(HoughLine {
                            start: [line[0] as f64, line[1] as f64],
                            end: [line[2] as f64, line[3] as f64],
                            confidence: 1.0,
                        });
                    }
                }
            }
        }

        let duration = start.elapsed();
        tracing::info!(
            target: "vectorize::performance",
            duration_ms = duration.as_millis(),
            lines_count = result.len(),
            "OpenCV 霍夫直线检测完成"
        );

        Ok(result)
    }

    /// 使用 OpenCV 查找轮廓
    #[tracing::instrument(name = "opencv_contours", skip(image), level = "debug")]
    pub fn find_contours_opencv(
        image: &GrayImage,
        min_length: usize,
    ) -> Result<Vec<Polyline>, String> {
        let start = std::time::Instant::now();
        // 转换为二值 Mat（反转：黑->白，白->黑，因为 findContours 查找白色对象）
        let mut mat = gray_image_to_binary_mat(image, true)?;

        let mut contours = opencv::core::Vector::<opencv::core::Vector<opencv::core::Point>>::new();

        // P11 锐评 v2.0 修复：移除不必要的 mut 警告
        #[allow(clippy::unnecessary_mut_passed)]
        imgproc::find_contours(
            &mut mat,
            &mut contours,
            imgproc::RETR_EXTERNAL,
            imgproc::CHAIN_APPROX_SIMPLE,
            Point::new(0, 0),
        )
        .map_err(|e| format!("查找轮廓失败：{}", e))?;

        let mut result = Vec::new();
        for i in 0..contours.len() {
            let contour = contours
                .get(i)
                .map_err(|e| format!("读取轮廓失败：{}", e))?;

            if contour.len() >= min_length {
                // 优化：批量转换为 slice
                let contour_slice = contour.as_slice();
                let polyline: Vec<[f64; 2]> = contour_slice
                    .iter()
                    .map(|pt| [pt.x as f64, pt.y as f64])
                    .collect();
                result.push(polyline);
            }
        }

        let duration = start.elapsed();
        tracing::info!(
            target: "vectorize::performance",
            duration_ms = duration.as_millis(),
            contours_count = result.len(),
            "OpenCV 轮廓提取完成"
        );

        Ok(result)
    }

    /// 使用 Douglas-Peucker 算法简化多边形（OpenCV approxPolyDP）
    ///
    /// # 参数
    /// * `polyline` - 输入多边形点集
    /// * `epsilon` - 简化精度（值越大简化程度越高）
    /// * `closed` - 是否为闭合多边形
    ///
    /// # 返回
    /// 简化后的多边形点集
    pub fn approx_polyline_opencv(
        polyline: &[Point2],
        epsilon: f64,
        closed: bool,
    ) -> Result<Vec<Point2>, String> {
        // 转换为 OpenCV Point 格式
        let points: Vector<Point> = polyline
            .iter()
            .map(|p| Point::new(p[0] as i32, p[1] as i32))
            .collect();

        let mut approx = Vector::<Point>::new();
        approx_poly_dp(&points, &mut approx, epsilon, closed)
            .map_err(|e| format!("approxPolyDP 失败：{}", e))?;

        // 转换回 Point2 格式
        let result: Vec<Point2> = approx.iter().map(|p| [p.x as f64, p.y as f64]).collect();

        Ok(result)
    }

    /// 简化轮廓（批量处理）
    pub fn simplify_contours_opencv(
        contours: &[Polyline],
        epsilon: f64,
    ) -> Result<Vec<Polyline>, String> {
        let mut result = Vec::with_capacity(contours.len());
        for contour in contours {
            match approx_polyline_opencv(contour, epsilon, true) {
                Ok(simplified) => {
                    // 至少保留 3 个点才能形成多边形
                    if simplified.len() >= 3 {
                        result.push(simplified);
                    }
                }
                Err(e) => {
                    tracing::warn!("简化轮廓失败：{}, 保留原始轮廓", e);
                    result.push(contour.clone());
                }
            }
        }
        Ok(result)
    }
}

#[cfg(feature = "opencv")]
pub use opencv_impl::*;

// ==================== 纯 Rust 实现 ====================

/// 边缘检测 - Sobel 算子（并行版本）
pub fn detect_edges(image: &GrayImage) -> GrayImage {
    let (width, height) = image.dimensions();

    // 预分配结果缓冲区
    let mut pixels = vec![0u8; (width * height) as usize];
    let width_usize = width as usize;
    let height_usize = height as usize;

    let sobel_x = [-1i32, 0, 1, -2, 0, 2, -1, 0, 1];
    let sobel_y = [-1i32, -2, -1, 0, 0, 0, 1, 2, 1];

    // 使用 slice 并行迭代器
    pixels
        .par_chunks_mut(width_usize)
        .enumerate()
        .skip(1)
        .take(height_usize - 2)
        .for_each(|(y, row)| {
            for (x, item) in row.iter_mut().enumerate().take(width_usize - 1).skip(1) {
                let mut gx = 0i32;
                let mut gy = 0i32;
                let mut idx = 0;

                for dy in -1isize..=1isize {
                    for dx in -1isize..=1isize {
                        let nx = (x as isize + dx) as usize;
                        let ny = (y as isize + dy) as usize;
                        let pixel = image.get_pixel(nx as u32, ny as u32)[0] as i32;
                        gx += pixel * sobel_x[idx];
                        gy += pixel * sobel_y[idx];
                        idx += 1;
                    }
                }

                let magnitude = ((gx * gx + gy * gy) as f32).sqrt() as u8;
                *item = magnitude;
            }
        });

    // 从原始像素数据构建图像
    GrayImage::from_raw(width, height, pixels).unwrap_or_else(|| GrayImage::new(width, height))
}

/// 二值化阈值处理（并行版本）
pub fn threshold(image: &GrayImage, thresh: u8) -> GrayImage {
    let mut result = GrayImage::new(image.width(), image.height());
    let width = image.width() as usize;

    // 使用 slice 并行迭代器
    result
        .as_mut()
        .par_chunks_mut(width)
        .zip(image.as_ref().par_chunks(width))
        .for_each(|(dst_row, src_row)| {
            for (dst_pixel, &src_pixel) in dst_row.iter_mut().zip(src_row.iter()) {
                *dst_pixel = if src_pixel >= thresh { 255u8 } else { 0u8 };
            }
        });

    result
}

/// 从边缘图像提取轮廓（迭代实现，避免栈溢出）
pub fn extract_contours(image: &GrayImage, min_length: usize) -> Vec<Polyline> {
    let (width, height) = image.dimensions();
    let mut visited = vec![vec![false; height as usize]; width as usize];
    let mut contours = Vec::new();

    for y in 0..height as usize {
        for x in 0..width as usize {
            if !visited[x][y] && image.get_pixel(x as u32, y as u32)[0] == 0 {
                let mut contour = Vec::new();
                dfs_contour_iterative(image, &mut visited, x as i32, y as i32, &mut contour);

                if contour.len() >= min_length {
                    contours.push(contour);
                }
            }
        }
    }

    contours
}

/// 迭代版本的 DFS 轮廓提取（使用显式栈避免栈溢出）
fn dfs_contour_iterative(
    image: &GrayImage,
    visited: &mut [Vec<bool>],
    start_x: i32,
    start_y: i32,
    contour: &mut Vec<Point2>,
) {
    let (width, height) = image.dimensions();

    // 使用显式栈模拟递归
    let mut stack: Vec<(i32, i32)> = Vec::new();
    stack.push((start_x, start_y));

    // 8 邻域方向
    let directions: [(i32, i32); 8] = [
        (-1, -1),
        (0, -1),
        (1, -1),
        (-1, 0),
        (1, 0),
        (-1, 1),
        (0, 1),
        (1, 1),
    ];

    while let Some((x, y)) = stack.pop() {
        // 边界检查
        if x < 0 || y < 0 || x >= width as i32 || y >= height as i32 {
            continue;
        }

        // 已访问检查
        if visited[x as usize][y as usize] {
            continue;
        }

        // 边缘检查（黑色像素为边缘）
        if image.get_pixel(x as u32, y as u32)[0] != 0 {
            continue;
        }

        // 标记为已访问并添加到轮廓
        visited[x as usize][y as usize] = true;
        contour.push([x as f64, y as f64]);

        // 将邻域点压入栈
        for (dx, dy) in &directions {
            let nx = x + dx;
            let ny = y + dy;

            if nx >= 0
                && ny >= 0
                && (nx < width as i32)
                && (ny < height as i32)
                && !visited[nx as usize][ny as usize]
            {
                stack.push((nx, ny));
            }
        }
    }
}

/// Douglas-Peucker 多边形简化算法
///
/// # 参数
/// * `points` - 输入点集
/// * `epsilon` - 简化精度（值越大简化程度越高）
///
/// # 返回
/// 简化后的点集
pub fn douglas_peucker(points: &[Point2], epsilon: f64) -> Vec<Point2> {
    if points.len() <= 2 {
        return points.to_vec();
    }

    let mut keep = vec![false; points.len()];
    keep[0] = true;
    keep[points.len() - 1] = true;

    douglas_peucker_recursive(points, 0, points.len() - 1, epsilon, &mut keep);

    points
        .iter()
        .enumerate()
        .filter(|(i, _)| keep[*i])
        .map(|(_, p)| *p)
        .collect()
}

/// Douglas-Peucker 递归实现
#[allow(clippy::needless_range_loop)]
fn douglas_peucker_recursive(
    points: &[Point2],
    start: usize,
    end: usize,
    epsilon: f64,
    keep: &mut [bool],
) {
    if start >= end {
        return;
    }

    // 计算最大距离
    let mut max_dist = 0.0;
    let mut max_idx = start;

    let line_start = points[start];
    let line_end = points[end];

    #[allow(clippy::needless_range_loop)]
    for i in (start + 1)..end {
        let dist = point_to_line_distance(points[i], line_start, line_end);
        if dist > max_dist {
            max_dist = dist;
            max_idx = i;
        }
    }

    // 递归细分
    if max_dist > epsilon {
        keep[max_idx] = true;
        douglas_peucker_recursive(points, start, max_idx, epsilon, keep);
        douglas_peucker_recursive(points, max_idx, end, epsilon, keep);
    }
}

/// 点到线段的距离
fn point_to_line_distance(point: Point2, line_start: Point2, line_end: Point2) -> f64 {
    let dx = line_end[0] - line_start[0];
    let dy = line_end[1] - line_start[1];

    if dx.abs() < 1e-10 && dy.abs() < 1e-10 {
        // 线段退化为点
        return ((point[0] - line_start[0]).powi(2) + (point[1] - line_start[1]).powi(2)).sqrt();
    }

    let t =
        ((point[0] - line_start[0]) * dx + (point[1] - line_start[1]) * dy) / (dx * dx + dy * dy);
    let t = t.clamp(0.0, 1.0);

    let proj_x = line_start[0] + t * dx;
    let proj_y = line_start[1] + t * dy;

    ((point[0] - proj_x).powi(2) + (point[1] - proj_y).powi(2)).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Luma;

    #[test]
    fn test_threshold() {
        let mut img = GrayImage::new(10, 10);
        img.put_pixel(5, 5, Luma([100]));
        img.put_pixel(6, 5, Luma([200]));

        let result = threshold(&img, 150);
        assert_eq!(result.get_pixel(5, 5)[0], 0);
        assert_eq!(result.get_pixel(6, 5)[0], 255);
    }

    #[test]
    fn test_detect_edges() {
        let mut img = GrayImage::new(10, 10);
        img.put_pixel(5, 5, Luma([0]));
        img.put_pixel(6, 5, Luma([255]));

        let edges = detect_edges(&img);
        assert!(edges.get_pixel(5, 5)[0] > 0 || edges.get_pixel(6, 5)[0] > 0);
    }
}
