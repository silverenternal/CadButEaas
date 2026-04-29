//! 文字标注分离 — 连通区域分析 + 启发式聚类
//!
//! 从二值图像中检测文字连通区域（blob），并启发式筛选文字特征。
//! 文字的启发式特征:
//! - 面积: min_area~max_area 像素（默认 5~5000，支持更小字符）
//! - 宽高比: 0.15~6.0（排除过长的线条和过大的区域，给宽字符更多容忍）
//! - 实心度: 面积/包围盒面积 > 0.2（扫描件噪声容忍度提高）
//! - 聚类: 邻近 blob 聚类为文字块，进一步确认文字区域

use common_types::Point2;
use image::{GrayImage, Luma};
use std::collections::HashMap;

/// 文字候选 blob（单个连通分量）
#[derive(Debug, Clone)]
pub struct TextBlob {
    /// 质心位置
    pub center: Point2,
    /// 包围盒左上角
    pub bbox_min: Point2,
    /// 包围盒右下角
    pub bbox_max: Point2,
    /// 像素数
    pub area: usize,
    /// 宽高比 (宽/高)
    pub aspect_ratio: f64,
    /// 实心度 (面积/包围盒面积)
    pub solidity: f64,
}

/// 文字块（聚类后的多个 blob）
#[derive(Debug, Clone)]
pub struct TextBlock {
    /// 合并后的包围盒左上角
    pub bbox_min: Point2,
    /// 合并后的包围盒右下角
    pub bbox_max: Point2,
    /// 包含的 blob 数量
    pub blob_count: usize,
    /// 总面积
    pub total_area: usize,
}

/// 包围盒
#[derive(Debug, Clone, Copy)]
pub struct BoundingBox {
    pub x_min: u32,
    pub y_min: u32,
    pub x_max: u32,
    pub y_max: u32,
}

impl BoundingBox {
    pub fn width(&self) -> u32 {
        self.x_max - self.x_min + 1
    }
    pub fn height(&self) -> u32 {
        self.y_max - self.y_min + 1
    }
    pub fn area(&self) -> u32 {
        self.width() * self.height()
    }
    pub fn center(&self) -> Point2 {
        [
            (self.x_min + self.x_max) as f64 / 2.0,
            (self.y_min + self.y_max) as f64 / 2.0,
        ]
    }
    pub fn intersection_area(&self, other: &BoundingBox) -> u32 {
        let x_min = self.x_min.max(other.x_min);
        let y_min = self.y_min.max(other.y_min);
        let x_max = self.x_max.min(other.x_max);
        let y_max = self.y_max.min(other.y_max);
        if x_max < x_min || y_max < y_min {
            return 0;
        }
        (x_max - x_min + 1) * (y_max - y_min + 1)
    }
    pub fn iou(&self, other: &BoundingBox) -> f64 {
        let inter = self.intersection_area(other) as f64;
        let union = (self.area() + other.area()) as f64 - inter;
        if union < 1e-10 {
            0.0
        } else {
            inter / union
        }
    }
}

/// 从二值图像中检测文字连通区域
///
/// # 参数
/// - `binary`: 二值图像（黑色=边缘/文字，白色=背景）
/// - `min_area`: 最小面积阈值
/// - `max_area`: 最大面积阈值
///
/// # 返回
/// 符合文字特征的 blob 列表（已聚类合并邻近字符）
///
/// 扫描图纸增强：
/// - 降低实心度要求，适应扫描噪声
/// - 支持更小字符（低至 3 像素）
/// - 放宽宽高比范围，适应中文字符
/// - 行优先聚类启发式，同一行文字更容易合并
pub fn detect_text_blobs(binary: &GrayImage, min_area: usize, max_area: usize) -> Vec<TextBlob> {
    let (width, height) = binary.dimensions();
    if width == 0 || height == 0 {
        return Vec::new();
    }

    // 两遍连通分量标记
    let labels = connected_components(binary, width, height);

    // 统计每个标签的像素数和包围盒
    let mut label_info: HashMap<u32, BlobStats> = HashMap::new();

    for y in 0..height {
        for x in 0..width {
            let label = labels[(y * width + x) as usize];
            if label == 0 {
                continue; // 背景
            }
            let entry = label_info.entry(label).or_insert_with(|| BlobStats {
                count: 0,
                x_min: x,
                y_min: y,
                x_max: x,
                y_max: y,
            });
            entry.count += 1;
            if x < entry.x_min {
                entry.x_min = x;
            }
            if x > entry.x_max {
                entry.x_max = x;
            }
            if y < entry.y_min {
                entry.y_min = y;
            }
            if y > entry.y_max {
                entry.y_max = y;
            }
        }
    }

    // 第一步：筛选候选 blob（宽松阈值，适应扫描件）
    let mut candidates = Vec::new();

    for stats in label_info.values() {
        let bbox = BoundingBox {
            x_min: stats.x_min,
            y_min: stats.y_min,
            x_max: stats.x_max,
            y_max: stats.y_max,
        };

        let area = stats.count;
        // 支持更小字符（扫描件可能小文字）
        if area < min_area.min(3) || area > max_area {
            continue;
        }

        let bbox_area = bbox.area();
        if bbox_area == 0 {
            continue;
        }

        let aspect_ratio = bbox.width() as f64 / bbox.height() as f64;
        // 进一步放宽宽高比范围：0.1 → 8.0，适应宽汉字和窄竖排文字
        if !(0.1..=8.0).contains(&aspect_ratio) {
            continue;
        }

        let solidity = area as f64 / bbox_area as f64;
        // 进一步降低实心度要求，扫描件噪声更多
        if solidity < 0.15 {
            continue;
        }

        // 额外启发式：排除极细的水平线（更可能是墙线）
        if bbox.height() <= 2 && bbox.width() > 20 {
            continue;
        }
        // 排除极细的垂直线
        if bbox.width() <= 2 && bbox.height() > 20 {
            continue;
        }

        let cx = (bbox.x_min + bbox.x_max) as f64 / 2.0;
        let cy = (bbox.y_min + bbox.y_max) as f64 / 2.0;

        candidates.push(TextBlob {
            center: [cx, cy],
            bbox_min: [bbox.x_min as f64, bbox.y_min as f64],
            bbox_max: [bbox.x_max as f64, bbox.y_max as f64],
            area,
            aspect_ratio,
            solidity,
        });
    }

    // 第二步：邻近聚类，合并同一块文字的多个字符
    // 对于扫描图纸，使用行启发式聚类
    let text_blocks = cluster_text_blobs_with_row_heuristic(&candidates, 2.0);

    // 第三步：从聚类结果生成最终文字块（返回合并后的包围盒）
    let mut result = Vec::new();
    for block in text_blocks {
        // 将整个文字块作为一个 blob 返回，便于擦除
        // 额外膨胀 1 像素，适应扫描件文字边缘模糊
        let mut min_x = block.bbox_min[0];
        let mut min_y = block.bbox_min[1];
        let mut max_x = block.bbox_max[0];
        let mut max_y = block.bbox_max[1];

        // 根据平均字符高度自适应膨胀
        let avg_h = (max_y - min_y) / block.blob_count as f64;
        let expand = avg_h.clamp(1.0, 3.0);
        min_x -= expand;
        min_y -= expand;
        max_x += expand;
        max_y += expand;

        // 裁剪到图像边界
        min_x = min_x.max(0.0);
        min_y = min_y.max(0.0);
        max_x = max_x.min((width - 1) as f64);
        max_y = max_y.min((height - 1) as f64);

        result.push(TextBlob {
            center: [(min_x + max_x) / 2.0, (min_y + max_y) / 2.0],
            bbox_min: [min_x, min_y],
            bbox_max: [max_x, max_y],
            area: block.total_area,
            aspect_ratio: (max_x - min_x) / (max_y - min_y),
            solidity: 0.5, // 聚类后的块实心度不再重要
        });
    }

    // 如果聚类结果为空，说明没有找到可聚类的文字，
    // 返回原始候选（可能是单个大字）
    if result.is_empty() {
        result = candidates;
    }

    // 最终过滤：移除太大的块（更可能是填充区域而不是文字）
    result.retain(|blob| {
        let area = (blob.bbox_max[0] - blob.bbox_min[0]) * (blob.bbox_max[1] - blob.bbox_min[1]);
        area < (max_area as f64) * 2.0
    });

    result
}

/// 聚类文字 blob：邻近的 blob 合并为文字块
///
/// 扫描图纸增强：行启发式 - 同一y范围内的 blob 更容易合并
fn cluster_text_blobs_with_row_heuristic(
    blobs: &[TextBlob],
    max_gap_factor: f64,
) -> Vec<TextBlock> {
    if blobs.is_empty() {
        return Vec::new();
    }

    // 计算平均字符高度
    let avg_height: f64 = blobs
        .iter()
        .map(|b| b.bbox_max[1] - b.bbox_min[1])
        .sum::<f64>()
        / blobs.len() as f64;

    // 距离阈值：基于平均字符高度
    let max_gap = avg_height * max_gap_factor;
    let max_gap_sq = max_gap * max_gap;

    // 行启发式：如果两个 blob 在同一行（y重叠），允许更大的 gap
    let row_overlap_threshold = avg_height * 0.3; // 30% 高度重叠就算同一行
    let row_max_gap = avg_height * max_gap_factor * 2.0; // 同一行允许两倍 gap
    let row_max_gap_sq = row_max_gap * row_max_gap;

    // 并查集聚类
    let mut union_find = UnionFind::new(blobs.len());

    for i in 0..blobs.len() {
        for j in (i + 1)..blobs.len() {
            let a = &blobs[i];
            let b = &blobs[j];

            // 检查y方向重叠（同一行）
            let a_y_min = a.bbox_min[1];
            let a_y_max = a.bbox_max[1];
            let b_y_min = b.bbox_min[1];
            let b_y_max = b.bbox_max[1];

            let intersection = (a_y_min.max(b_y_min))..(a_y_max.min(b_y_max));
            let has_overlap = intersection.start < intersection.end;
            let overlap_len = if has_overlap {
                (intersection.end - intersection.start)
                    / ((a_y_max - a_y_min + b_y_max - b_y_min) / 2.0)
            } else {
                0.0
            };

            // 根据是否同一行选择不同阈值
            let threshold_sq = if overlap_len > row_overlap_threshold {
                row_max_gap_sq
            } else {
                max_gap_sq
            };

            if blob_distance_squared(a, b) < threshold_sq {
                union_find.union(i, j);
            }
        }
    }

    // 按根分组
    let mut groups: HashMap<usize, Vec<&TextBlob>> = HashMap::new();
    for (idx, blob) in blobs.iter().enumerate() {
        let root = union_find.find(idx);
        groups.entry(root).or_default().push(blob);
    }

    // 合并每组为一个 TextBlock
    let mut blocks = Vec::new();
    for group in groups.values() {
        let mut min_x = f64::INFINITY;
        let mut min_y = f64::INFINITY;
        let mut max_x = -f64::INFINITY;
        let mut max_y = -f64::INFINITY;
        let mut total_area = 0;

        for blob in group {
            min_x = min_x.min(blob.bbox_min[0]);
            min_y = min_y.min(blob.bbox_min[1]);
            max_x = max_x.max(blob.bbox_max[0]);
            max_y = max_y.max(blob.bbox_max[1]);
            total_area += blob.area;
        }

        blocks.push(TextBlock {
            bbox_min: [min_x, min_y],
            bbox_max: [max_x, max_y],
            blob_count: group.len(),
            total_area,
        });
    }

    blocks
}

/// 计算两个 blob 之间的最近距离平方
fn blob_distance_squared(a: &TextBlob, b: &TextBlob) -> f64 {
    // 检查包围盒是否重叠
    if a.bbox_max[0] >= b.bbox_min[0]
        && b.bbox_max[0] >= a.bbox_min[0]
        && a.bbox_max[1] >= b.bbox_min[1]
        && b.bbox_max[1] >= a.bbox_min[1]
    {
        return 0.0; // 重叠，距离为零
    }

    // 计算最近点距离平方
    let dx = if a.bbox_max[0] < b.bbox_min[0] {
        b.bbox_min[0] - a.bbox_max[0]
    } else if b.bbox_max[0] < a.bbox_min[0] {
        a.bbox_min[0] - b.bbox_max[0]
    } else {
        0.0
    };

    let dy = if a.bbox_max[1] < b.bbox_min[1] {
        b.bbox_min[1] - a.bbox_max[1]
    } else if b.bbox_max[1] < a.bbox_min[1] {
        a.bbox_min[1] - b.bbox_max[1]
    } else {
        0.0
    };

    dx * dx + dy * dy
}

/// 连通分量标记统计
struct BlobStats {
    count: usize,
    x_min: u32,
    y_min: u32,
    x_max: u32,
    y_max: u32,
}

/// 两遍连通分量标记（4-邻域）
/// 返回与图像同尺寸的标签数组，0=背景，>0=标签
fn connected_components(binary: &GrayImage, width: u32, height: u32) -> Vec<u32> {
    let mut labels = vec![0u32; (width * height) as usize];
    let mut next_label: u32 = 1;
    let mut union_find = UnionFind::new((width * height) as usize);

    let idx = |x: u32, y: u32| (y * width + x) as usize;

    // 第一遍: 扫描并分配标签
    for y in 0..height {
        for x in 0..width {
            let is_foreground = binary.get_pixel(x, y)[0] == 0; // 黑色=前景
            if !is_foreground {
                continue;
            }

            // 检查 4-邻域（左、上）
            let neighbors: Vec<u32> = {
                let mut n = Vec::with_capacity(2);
                if x > 0 {
                    let left_label = labels[idx(x - 1, y)];
                    if left_label > 0 {
                        n.push(left_label);
                    }
                }
                if y > 0 {
                    let up_label = labels[idx(x, y - 1)];
                    if up_label > 0 {
                        n.push(up_label);
                    }
                }
                n
            };

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

    // 第二遍: 解析等价关系
    for i in 0..labels.len() {
        if labels[i] > 0 {
            labels[i] = union_find.find(labels[i] as usize) as u32;
        }
    }

    // 重新编号为连续标签
    let mut label_map: std::collections::HashMap<u32, u32> = std::collections::HashMap::new();
    let mut counter: u32 = 1;
    for label in &labels {
        if *label > 0 && !label_map.contains_key(label) {
            label_map.insert(*label, counter);
            counter += 1;
        }
    }
    for label in &mut labels {
        if *label > 0 {
            *label = *label_map.get(label).unwrap();
        }
    }

    labels
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

/// 擦除文字 blob 区域（填充为白色/255）
///
/// 将 blob 的包围盒内所有前景像素设为背景色，
/// 膨胀一个像素确保文字边缘完全清除，
/// 防止后续边缘检测将其提取为线段。
pub fn erase_text_blobs(binary: &GrayImage, blobs: &[TextBlob]) -> GrayImage {
    let mut result = binary.clone();
    let (width, height) = result.dimensions();

    for blob in blobs {
        // 膨胀 2 像素，确保完全覆盖文字边缘（扫描件可能有偏移）
        let x_min = (blob.bbox_min[0] - 2.0).floor().max(0.0) as u32;
        let y_min = (blob.bbox_min[1] - 2.0).floor().max(0.0) as u32;
        let x_max = (blob.bbox_max[0] + 2.0).ceil().min((width - 1) as f64) as u32;
        let y_max = (blob.bbox_max[1] + 2.0).ceil().min((height - 1) as f64) as u32;

        for y in y_min..=y_max {
            for x in x_min..=x_max {
                result.put_pixel(x, y, Luma([255]));
            }
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_image(width: u32, height: u32, blobs: &[(u32, u32, u32, u32)]) -> GrayImage {
        let mut img = GrayImage::from_pixel(width, height, Luma([255])); // 白色背景
        for (x_min, y_min, x_max, y_max) in blobs {
            for y in *y_min..=*y_max {
                for x in *x_min..=*x_max {
                    img.put_pixel(x, y, Luma([0])); // 黑色 blob
                }
            }
        }
        img
    }

    #[test]
    fn test_detect_text_blob_simple() {
        // 3×5 的黑色矩形 blob（模拟文字字符）
        let img = create_test_image(20, 20, &[(2, 3, 4, 7)]);
        let blobs = detect_text_blobs(&img, 5, 5000);

        assert_eq!(blobs.len(), 1);
        // 原始: x 2-4 (3px), y 3-7 (5px)
        // 最终聚类膨胀 ±1px: x 1-5 (w=5px), y 2-8 (h=7px)
        // aspect ratio = 5/7 ≈ 0.714
        assert!(blobs[0].area >= 12); // 至少包含原始 15 像素
        let expected_ratio = 5.0 / 7.0; // ≈ 0.714
        assert!((blobs[0].aspect_ratio - expected_ratio).abs() < 0.1);
    }

    #[test]
    fn test_filter_long_lines() {
        // 长而细的 blob（水平线条）→ 不被识别为文字
        let img = create_test_image(100, 100, &[(0, 10, 80, 10)]);
        let blobs = detect_text_blobs(&img, 5, 5000);

        // 宽高比 = 81/1 = 81 > 5.0 → 应被过滤
        assert!(blobs.is_empty(), "长而细的 blob 不应被识别为文字");
    }

    #[test]
    fn test_filter_large_area() {
        // 大面积 blob（80×80 = 6400 像素）→ 不被识别为文字
        let img = create_test_image(100, 100, &[(10, 10, 89, 89)]);
        let blobs = detect_text_blobs(&img, 10, 5000);

        assert!(blobs.is_empty(), "大面积 blob 不应被识别为文字");
    }

    #[test]
    fn test_filter_small_area() {
        // 极小 blob（1×2）→ 面积太小
        let img = create_test_image(20, 20, &[(5, 5, 5, 6)]);
        let blobs = detect_text_blobs(&img, 5, 5000);

        assert!(blobs.is_empty(), "面积极小的 blob 不应被识别为文字");
    }

    #[test]
    fn test_erase_text_blobs() {
        let img = create_test_image(20, 20, &[(5, 5, 8, 8)]);
        let blobs = detect_text_blobs(&img, 5, 5000);
        assert_eq!(blobs.len(), 1);

        let erased = erase_text_blobs(&img, &blobs);

        // 检查 blob 区域是否被填充为白色
        for y in 5..=8 {
            for x in 5..=8 {
                assert_eq!(erased.get_pixel(x, y)[0], 255, "blob 区域应被填充为白色");
            }
        }
    }

    #[test]
    fn test_multiple_blobs_clustered() {
        // 3 个文字 blob + 1 个线条（应被过滤）
        // 三个字符在同一行且邻近，会被聚类为一个文字块
        let img = create_test_image(
            100,
            50,
            &[
                (5, 5, 10, 12),  // blob 1: 6×8
                (20, 5, 25, 10), // blob 2: 6×6
                (35, 5, 42, 11), // blob 3: 8×7
                (0, 40, 99, 40), // 水平线条（应被过滤）
            ],
        );
        let blobs = detect_text_blobs(&img, 5, 5000);

        // 三个邻近字符会被聚类为一个文字块，这是期望行为
        assert_eq!(blobs.len(), 1, "三个邻近文字应聚类为 1 个文字块");
        assert!(blobs[0].bbox_max[0] > 35.0); // 覆盖最右侧字符
        assert!(blobs[0].bbox_min[0] < 10.0); // 覆盖最左侧字符
    }

    #[test]
    fn test_multiple_blobs_separated() {
        // 3 个文字 blob 分隔较远，保持独立
        let img = create_test_image(
            200,
            100,
            &[
                (10, 10, 20, 20),   // blob 1
                (60, 10, 70, 20),   // blob 2 (分隔较远)
                (110, 10, 120, 20), // blob 3 (分隔较远)
                (0, 80, 199, 80),   // 水平线条（应被过滤）
            ],
        );
        let blobs = detect_text_blobs(&img, 5, 5000);

        // 分隔较远仍然保持三个独立 blob
        assert_eq!(blobs.len(), 3, "分隔较远的文字应保持为 3 个独立块");
    }

    #[test]
    fn test_empty_image() {
        let img = GrayImage::from_pixel(20, 20, Luma([255])); // 全白背景
        let blobs = detect_text_blobs(&img, 5, 5000);
        assert!(blobs.is_empty());
    }

    #[test]
    fn test_zero_dimensions() {
        let blobs = detect_text_blobs(&GrayImage::new(0, 0), 5, 5000);
        assert!(blobs.is_empty());
    }
}
