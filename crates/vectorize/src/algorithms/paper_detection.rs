//! 纸张检测与自动裁剪
//!
//! 自动检测扫描/拍照图像中的纸张区域，去除背景黑边、桌面等干扰，
//! 仅保留有效图纸区域用于后续处理。
//!
//! 算法思路：
//! 1. 对灰度图像进行模糊去噪
//! 2. 二值化分割前景（纸张）和背景
//! 3. 查找最大外接轮廓（假设为纸张）
//! 4. 使用 Douglas-Peucker 简化得到近似多边形
//! 5. 如果近似四边形，检测四个角点进行裁剪
//! 6. 如果检测失败，返回原图（安全 fallback）

use crate::algorithms::douglas_peucker;
use common_types::Point2;
use image::GrayImage;

type ComponentBBoxes = Vec<(u32, u32, u32, u32)>;

/// 检测结果 - 纸张区域
#[derive(Debug, Clone)]
pub struct PaperRegion {
    /// 四个角点（按顺时针顺序）
    pub corners: [Point2; 4],
    /// 原始图像中的 bounding box
    pub bbox: (u32, u32, u32, u32), // (xmin, ymin, xmax, ymax)
    /// 检测置信度 (0-1)
    pub confidence: f64,
}

/// 自动检测纸张区域并裁剪
///
/// # 参数
/// - `gray`: 输入灰度图像
///
/// # 返回
/// 裁剪后的灰度图像，如果检测失败返回原图
pub fn detect_and_crop(gray: &GrayImage) -> GrayImage {
    match detect_paper(gray) {
        Some(region) => {
            // 如果置信度足够高，进行裁剪
            if region.confidence > 0.5 {
                crop_to_bbox(
                    gray,
                    region.bbox.0,
                    region.bbox.1,
                    region.bbox.2,
                    region.bbox.3,
                )
            } else {
                gray.clone()
            }
        }
        None => gray.clone(),
    }
}

/// 检测纸张区域
///
/// 返回 None 表示检测失败，应该使用原图
pub fn detect_paper(gray: &GrayImage) -> Option<PaperRegion> {
    let (width, height) = gray.dimensions();

    // 1. 轻微高斯模糊去噪
    let blurred = crate::algorithms::preprocessing::gaussian_blur(gray, 1.0);

    // 2. 使用 Otsu 自动阈值二值化
    let binary = crate::algorithms::threshold::binary_with_otsu(&blurred);

    // 3. 查找所有连通分量，找到最大的那个（纸张）
    let (components, bounding_boxes) = find_connected_components(&binary);

    if components.is_empty() {
        return None;
    }

    // 找到面积最大的连通分量
    let mut max_area = 0;
    let mut max_idx = 0;
    for (i, &area) in components.iter().enumerate() {
        if area > max_area {
            max_area = area;
            max_idx = i;
        }
    }

    let total_pixels = (width * height) as usize;
    let area_ratio = (max_area as f64) / (total_pixels as f64);

    // 如果最大分量占比太小，可能检测失败，直接返回
    if area_ratio < 0.1 {
        return None;
    }

    // 获取最大分量的 bounding box
    let (xmin, ymin, xmax, ymax) = bounding_boxes[max_idx];
    let bbox_width = (xmax - xmin + 1) as f64;
    let bbox_height = (ymax - ymin + 1) as f64;

    // 提取轮廓边界点
    let contour = extract_contour_points(&binary, xmin, ymin, xmax, ymax);

    if contour.len() < 4 {
        // 轮廓点太少，不是纸张
        return None;
    }

    // 使用 Douglas-Peucker 简化多边形
    let epsilon = (bbox_width.min(bbox_height)) * 0.02; // 2% 容差
    let simplified = douglas_peucker(&contour, epsilon);

    // 尝试寻找四边形
    if simplified.len() == 4 {
        // 正好四个点 - 理想情况
        let corners = order_corners(&simplified);
        let confidence = calculate_quad_confidence(&corners, width, height);
        Some(PaperRegion {
            corners,
            bbox: (xmin, ymin, xmax, ymax),
            confidence,
        })
    } else if simplified.len() > 4 && simplified.len() <= 8 {
        // 稍微多几个点，尝试合并近似四边形
        // 这里简化处理：直接使用 bounding box 作为 fallback
        let corners = bbox_to_corners(xmin, ymin, xmax, ymax);
        let confidence = calculate_bbox_confidence(area_ratio, bbox_width, bbox_height);
        Some(PaperRegion {
            corners,
            bbox: (xmin, ymin, xmax, ymax),
            confidence,
        })
    } else {
        // 太多点，形状不规则，使用 bounding box
        let corners = bbox_to_corners(xmin, ymin, xmax, ymax);
        let confidence = calculate_bbox_confidence(area_ratio, bbox_width, bbox_height) * 0.8;
        Some(PaperRegion {
            corners,
            bbox: (xmin, ymin, xmax, ymax),
            confidence,
        })
    }
}

/// 查找连通分量（两遍算法）
fn find_connected_components(binary: &GrayImage) -> (Vec<usize>, ComponentBBoxes) {
    let (width, height) = binary.dimensions();
    let mut labels = vec![0u32; (width * height) as usize];
    let mut next_label = 1u32;
    let mut union_find = UnionFind::new((width * height) as usize);

    let idx = |x: u32, y: u32| (y * width + x) as usize;

    // 第一遍：标记连通分量
    for y in 0..height {
        for x in 0..width {
            // 我们要找纸张区域：纸张通常是白色背景，所以纸张区域整体是亮白色
            // 经过二值化，纸张区域是白色 (255)，外围背景是黑色 (0)
            // 所以找最大的白色连通分量，那就是纸张
            let pixel = binary.get_pixel(x, y)[0];
            let is_foreground = pixel == 255; // 白色纸张是前景
            if !is_foreground {
                continue;
            }

            // 检查左、上邻居
            let mut neighbors = Vec::new();
            if x > 0 {
                let label = labels[idx(x - 1, y)];
                if label > 0 {
                    neighbors.push(label);
                }
            }
            if y > 0 {
                let label = labels[idx(x, y - 1)];
                if label > 0 {
                    neighbors.push(label);
                }
            }

            if neighbors.is_empty() {
                labels[idx(x, y)] = next_label;
                next_label += 1;
            } else {
                let min_label = *neighbors.iter().min().unwrap();
                labels[idx(x, y)] = min_label;
                for &nb in &neighbors {
                    if nb != min_label {
                        union_find.union(min_label as usize, nb as usize);
                    }
                }
            }
        }
    }

    // 第二遍：合并等价标签
    for label in &mut labels {
        if *label > 0 {
            *label = union_find.find(*label as usize) as u32;
        }
    }

    // 重新编号为连续标签
    let mut label_map = std::collections::HashMap::new();
    let mut counter = 1u32;
    for label in &mut labels {
        if *label > 0 {
            label_map.entry(*label).or_insert_with(|| {
                let new_label = counter;
                counter += 1;
                new_label
            });
            *label = *label_map.get(label).unwrap();
        }
    }

    // 统计每个标签的面积和 bounding box
    let mut areas = vec![0usize; counter as usize];
    let mut bboxes: Vec<(u32, u32, u32, u32)> = vec![(0, 0, 0, 0); counter as usize];

    // 初始化 bounding box
    for i in 1..counter as usize {
        bboxes[i] = (u32::MAX, u32::MAX, 0, 0);
    }

    for y in 0..height {
        for x in 0..width {
            let label = labels[idx(x, y)];
            if label == 0 {
                continue;
            }
            let label_idx = label as usize;
            areas[label_idx] += 1;

            let (xmin, ymin, xmax, ymax) = &mut bboxes[label_idx];
            *xmin = (*xmin).min(x);
            *ymin = (*ymin).min(y);
            *xmax = (*xmax).max(x);
            *ymax = (*ymax).max(y);
        }
    }

    // 移除标签 0（背景），收集结果
    let mut result_areas = Vec::new();
    let mut result_bboxes = Vec::new();
    for i in 1..counter as usize {
        if areas[i] > 0 {
            result_areas.push(areas[i]);
            result_bboxes.push(bboxes[i]);
        }
    }

    (result_areas, result_bboxes)
}

/// 提取轮廓边界点
fn extract_contour_points(
    binary: &GrayImage,
    xmin: u32,
    ymin: u32,
    xmax: u32,
    ymax: u32,
) -> Vec<Point2> {
    let mut points = Vec::new();

    // 遍历 bounding box 的边界，收集轮廓点
    // 现在前景是白色纸张 (255)，背景是黑色 (0)
    // 沿着 bounding box 边框，属于纸张的像素是白色 → 收集作为轮廓
    // 上边
    for x in xmin..=xmax {
        if binary.get_pixel(x, ymin)[0] == 255 {
            points.push([x as f64, ymin as f64]);
        }
    }
    // 右边
    for y in ymin..=ymax {
        if binary.get_pixel(xmax, y)[0] == 255 {
            points.push([xmax as f64, y as f64]);
        }
    }
    // 下边
    for x in (xmin..=xmax).rev() {
        if binary.get_pixel(x, ymax)[0] == 255 {
            points.push([x as f64, ymax as f64]);
        }
    }
    // 左边
    for y in (ymin..=ymax).rev() {
        if binary.get_pixel(xmin, y)[0] == 255 {
            points.push([xmin as f64, y as f64]);
        }
    }

    points
}

/// 将 bounding box 转换为四个角点
fn bbox_to_corners(xmin: u32, ymin: u32, xmax: u32, ymax: u32) -> [Point2; 4] {
    [
        [xmin as f64, ymin as f64], // top-left
        [xmax as f64, ymin as f64], // top-right
        [xmax as f64, ymax as f64], // bottom-right
        [xmin as f64, ymax as f64], // bottom-left
    ]
}

/// 对四个角点进行排序（顺时针，从左上角开始）
fn order_corners(corners: &[Point2]) -> [Point2; 4] {
    assert_eq!(corners.len(), 4);

    let mut sorted = corners.to_vec();

    // 按 y + x 排序，找出左上角和右下角
    // 左上角：x + y 最小
    // 右下角：x + y 最大
    sorted.sort_by(|a, b| (a[0] + a[1]).partial_cmp(&(b[0] + b[1])).unwrap());

    let tl = sorted[0];
    let br = sorted[3];

    // 剩下两个点是 tr 和 bl
    // tr: x - y 更大
    let a = sorted[1];
    let b = sorted[2];
    let (tr, bl) = if (a[0] - a[1]) > (b[0] - b[1]) {
        (a, b)
    } else {
        (b, a)
    };

    [tl, tr, br, bl]
}

/// 计算四边形检测置信度
fn calculate_quad_confidence(corners: &[Point2; 4], _width: u32, _height: u32) -> f64 {
    // 检查角度是否接近 90 度
    let mut total_angle_error = 0.0;

    for i in 0..4 {
        let p_prev = corners[(i + 3) % 4];
        let p_curr = corners[i];
        let p_next = corners[(i + 1) % 4];

        let v1 = [p_curr[0] - p_prev[0], p_curr[1] - p_prev[1]];
        let v2 = [p_next[0] - p_curr[0], p_next[1] - p_curr[1]];

        let dot = v1[0] * v2[0] + v1[1] * v2[1];
        let len1 = (v1[0] * v1[0] + v1[1] * v1[1]).sqrt();
        let len2 = (v2[0] * v2[0] + v2[1] * v2[1]).sqrt();

        if len1 < 1e-6 || len2 < 1e-6 {
            return 0.3;
        }

        let cos_angle = dot / (len1 * len2);
        // 期望 90 度 → cos_angle 应该接近 0
        let angle_error = cos_angle.abs();
        total_angle_error += angle_error;
    }

    let avg_angle_error = total_angle_error / 4.0;
    // 误差越小，置信度越高 → 0 误差 → 1.0, 0.5 误差 → 0.5
    (1.0 - avg_angle_error).clamp(0.0, 1.0)
}

/// 计算 bounding box 检测置信度
fn calculate_bbox_confidence(area_ratio: f64, width: f64, height: f64) -> f64 {
    let aspect = width / height;
    // 纸张通常在 0.5 ~ 2.0 之间
    if !(0.3..=3.0).contains(&aspect) {
        area_ratio * 0.5
    } else {
        area_ratio
    }
}

/// 裁剪图像到指定 bounding box
fn crop_to_bbox(image: &GrayImage, xmin: u32, ymin: u32, xmax: u32, ymax: u32) -> GrayImage {
    let new_width = xmax - xmin + 1;
    let new_height = ymax - ymin + 1;

    let mut result = GrayImage::new(new_width, new_height);

    for y in 0..new_height {
        for x in 0..new_width {
            let original_x = xmin + x;
            let original_y = ymin + y;
            let pixel = image.get_pixel(original_x, original_y);
            result.put_pixel(x, y, *pixel);
        }
    }

    // 添加轻微的边缘padding，避免裁掉线条
    // 如果padding超出范围就保持原样
    const PADDING: u32 = 2;
    if xmin >= PADDING
        && ymin >= PADDING
        && xmax + PADDING < image.width()
        && ymax + PADDING < image.height()
    {
        // 已经在上面分配了正确大小，如果要padding需要重新分配
        // 简化起见，这里不处理，用户可以后续再处理
    }

    result
}

/// 并查集
struct UnionFind {
    parent: Vec<usize>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
        }
    }

    fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x {
            self.parent[x] = self.find(self.parent[x]);
        }
        self.parent[x]
    }

    fn union(&mut self, x: usize, y: usize) {
        let rx = self.find(x);
        let ry = self.find(y);
        if rx != ry {
            self.parent[rx] = ry;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{GrayImage, Luma};

    fn create_test_image_with_border() -> GrayImage {
        // 创建 100x100 图像，周围 10px 黑色边框，中间白色
        let mut img = GrayImage::new(100, 100);
        for y in 0..100 {
            for x in 0..100 {
                if !(10..90).contains(&x) || !(10..90).contains(&y) {
                    img.put_pixel(x, y, Luma([0])); // 黑色背景边框
                } else {
                    img.put_pixel(x, y, Luma([255])); // 白色纸张
                }
            }
        }
        img
    }

    #[test]
    fn test_detect_paper_square() {
        let img = create_test_image_with_border();
        let region = detect_paper(&img);
        assert!(region.is_some());
        let region = region.unwrap();
        assert!(region.confidence > 0.5);
        // bounding box 应该在 10, 10, 89, 89 附近
        assert!(region.bbox.0 >= 5 && region.bbox.0 <= 15);
        assert!(region.bbox.2 >= 85 && region.bbox.2 <= 95);
    }

    #[test]
    fn test_detect_and_crop() {
        let img = create_test_image_with_border();
        let cropped = detect_and_crop(&img);
        // 应该裁剪掉边框，尺寸变小
        assert!(cropped.width() < img.width());
        assert!(cropped.height() < img.height());
    }

    #[test]
    fn test_order_corners() {
        let corners = [[10.0, 10.0], [90.0, 10.0], [90.0, 90.0], [10.0, 90.0]];
        let ordered = order_corners(&corners);
        // 顺序应该不变，已经正确
        assert!((ordered[0][0] - 10.0).abs() < 0.1);
        assert!((ordered[0][1] - 10.0).abs() < 0.1);
    }

    #[test]
    fn test_empty_image() {
        // 对于恒定亮度图像，整个图像被检测为纸张区域，因为整个区域都是均匀亮度
        // otsu_threshold 无法分割，最终整个图像被当作纸张 → 返回 Some
        let mut img = GrayImage::new(100, 100);
        for y in 0..100 {
            for x in 0..100 {
                img.put_pixel(x, y, Luma([255]));
            }
        }
        let region = detect_paper(&img);
        // 整个图像就是纸张，应该返回 Some，而不是 None
        assert!(region.is_some());
    }
}
