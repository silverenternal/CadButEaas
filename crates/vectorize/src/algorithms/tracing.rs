//! 智能轨迹追踪算法
//!
//! 基于骨架图的轨迹追踪，交点分类和 strokes 提取
//! - 8-邻域像素的交点分类（端点、L型、T型、X型、多分支）
//! - BFS 遍历提取 strokes
//! - 闭环检测

use super::SubpixelPoint;
use image::GrayImage;
use std::collections::{HashMap, HashSet, VecDeque};

/// 交点类型
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum JunctionType {
    /// 端点（只有1个邻居）
    Endpoint,
    /// L型（2个不连续邻居）
    LShape,
    /// T型（3个邻居）
    TShape,
    /// X型（4个邻居）
    XShape,
    /// 多分支（>4个邻居）
    Multi,
}

/// 交点信息
#[derive(Debug, Clone)]
pub struct Junction {
    /// 像素级 X 坐标
    pub x: u32,
    /// 像素级 Y 坐标
    pub y: u32,
    /// 亚像素精度坐标
    pub precise: SubpixelPoint,
    /// 交点类型
    pub junction_type: JunctionType,
    /// 分支方向编码（0-7，顺时针）
    pub directions: Vec<u8>,
}

/// 笔画（连续线段）
#[derive(Debug, Clone)]
pub struct Stroke {
    /// 起点（亚像素）
    pub start: SubpixelPoint,
    /// 终点（亚像素）
    pub end: SubpixelPoint,
    /// 中间点序列
    pub points: Vec<SubpixelPoint>,
    /// 总长度（像素）
    pub length: f64,
    /// 是否为闭环
    pub is_closed: bool,
}

impl Stroke {
    /// 计算总长度
    pub fn calculate_length(&self) -> f64 {
        if self.points.is_empty() {
            return self.start.distance(&self.end);
        }
        let mut len = 0.0;
        let mut prev = &self.start;
        for p in &self.points {
            len += prev.distance(p);
            prev = p;
        }
        len += prev.distance(&self.end);
        len
    }
}

// 8-邻域偏移（顺时针顺序：从正北开始）
//   7  0  1
//   6  *  2
//   5  4  3
const NEIGHBORS: [(i32, i32); 8] = [
    (0, -1),  // 0: N
    (1, -1),  // 1: NE
    (1, 0),   // 2: E
    (1, 1),   // 3: SE
    (0, 1),   // 4: S
    (-1, 1),  // 5: SW
    (-1, 0),  // 6: W
    (-1, -1), // 7: NW
];

// 相反方向映射
const OPPOSITE: [u8; 8] = [4, 5, 6, 7, 0, 1, 2, 3];

/// 检测图像中的所有交点
pub fn detect_junctions(image: &GrayImage) -> Vec<Junction> {
    let (width, height) = image.dimensions();
    let mut junctions = Vec::new();

    for y in 0..height {
        for x in 0..width {
            let pixel = image.get_pixel(x, y)[0];
            if pixel == 0 {
                continue; // 跳过非骨架像素
            }

            let mut neighbors = Vec::new();
            for (dir, &(dx, dy)) in NEIGHBORS.iter().enumerate() {
                let nx = x as i32 + dx;
                let ny = y as i32 + dy;
                if nx >= 0 && nx < width as i32 && ny >= 0 && ny < height as i32 {
                    let n_pixel = image.get_pixel(nx as u32, ny as u32)[0];
                    if n_pixel > 0 {
                        neighbors.push(dir as u8);
                    }
                }
            }

            let junction_type = match neighbors.len() {
                0 => continue,               // 孤立像素
                1 => JunctionType::Endpoint, // 端点
                2 => {
                    // 判断两个方向是否相邻（非L型即直线）
                    let diff = ((neighbors[1] as i32 - neighbors[0] as i32) + 8) % 8;
                    if diff == 1 || diff == 7 {
                        continue; // 相邻方向，是普通直线点，跳过
                    } else {
                        JunctionType::LShape // L型
                    }
                }
                3 => JunctionType::TShape,
                4 => JunctionType::XShape,
                _ => JunctionType::Multi,
            };

            // 亚像素精修（5x5 窗口质心计算）
            let precise = refine_subpixel_centroid(image, x, y);

            junctions.push(Junction {
                x,
                y,
                precise,
                junction_type,
                directions: neighbors,
            });
        }
    }

    junctions
}

/// 5x5 窗口灰度质心精修
fn refine_subpixel_centroid(image: &GrayImage, cx: u32, cy: u32) -> SubpixelPoint {
    let (width, height) = image.dimensions();
    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    let mut sum_w = 0.0;

    for dy in -2..=2 {
        for dx in -2..=2 {
            let px = cx as i32 + dx;
            let py = cy as i32 + dy;
            if px >= 0 && px < width as i32 && py >= 0 && py < height as i32 {
                let w = image.get_pixel(px as u32, py as u32)[0] as f64;
                sum_x += dx as f64 * w;
                sum_y += dy as f64 * w;
                sum_w += w;
            }
        }
    }

    if sum_w > 0.0 {
        SubpixelPoint {
            x: cx as f64 + sum_x / sum_w,
            y: cy as f64 + sum_y / sum_w,
        }
    } else {
        SubpixelPoint {
            x: cx as f64,
            y: cy as f64,
        }
    }
}

/// 从骨架提取所有笔画
pub fn extract_strokes(image: &GrayImage, junctions: &[Junction]) -> Vec<Stroke> {
    let (width, height) = image.dimensions();
    let mut strokes = Vec::new();
    let mut visited = HashSet::new();

    // 构建交点位置映射
    let mut junction_map = HashMap::new();
    for junc in junctions {
        junction_map.insert((junc.x, junc.y), junc);
    }

    // 从每个交点的每个方向开始追踪
    for junc in junctions {
        for &dir in &junc.directions {
            let key = ((junc.x, junc.y), dir);
            if visited.contains(&key) {
                continue;
            }

            if let Some(stroke) = trace_stroke_from(
                image,
                (junc.x, junc.y),
                dir,
                &junction_map,
                &mut visited,
                width,
                height,
            ) {
                strokes.push(stroke);
            }
        }
    }

    // 检测并提取闭环（没有端点的循环）
    let closed_strokes = extract_closed_loops(image, &junction_map, &mut visited, width, height);
    strokes.extend(closed_strokes);

    strokes
}

/// 从交点沿指定方向追踪笔画
#[allow(clippy::too_many_arguments)]
fn trace_stroke_from(
    image: &GrayImage,
    start: (u32, u32),
    start_dir: u8,
    junction_map: &HashMap<(u32, u32), &Junction>,
    visited: &mut HashSet<((u32, u32), u8)>,
    width: u32,
    height: u32,
) -> Option<Stroke> {
    let mut points = Vec::new();
    let mut cx = start.0 as i32 + NEIGHBORS[start_dir as usize].0;
    let mut cy = start.1 as i32 + NEIGHBORS[start_dir as usize].1;
    let mut came_from = OPPOSITE[start_dir as usize];

    // 标记起点方向已访问
    visited.insert((start, start_dir));

    while cx >= 0 && cx < width as i32 && cy >= 0 && cy < height as i32 {
        let curr = (cx as u32, cy as u32);

        // 检查是否到达另一个交点
        if let Some(junc) = junction_map.get(&curr) {
            // 标记到达方向已访问
            visited.insert((curr, came_from));

            let start_junc = junction_map.get(&start).unwrap();

            // 计算起点和终点的亚像素坐标
            let start_point = start_junc.precise;
            let end_point = junc.precise;

            let mut stroke = Stroke {
                start: start_point,
                end: end_point,
                points: Vec::new(),
                length: 0.0,
                is_closed: false,
            };

            // 转换中间点为亚像素
            stroke.points = points
                .iter()
                .map(|&(x, y)| refine_subpixel_centroid(image, x, y))
                .collect();

            stroke.length = stroke.calculate_length();
            return Some(stroke);
        }

        // 如果是骨架像素，继续追踪
        let pixel = image.get_pixel(curr.0, curr.1)[0];
        if pixel == 0 {
            break;
        }

        points.push(curr);

        // 找下一个方向（排除来路）
        let mut next_dir = None;
        for (dir, &(dx, dy)) in NEIGHBORS.iter().enumerate() {
            if dir as u8 == came_from {
                continue;
            }
            let nx = cx + dx;
            let ny = cy + dy;
            if nx >= 0 && nx < width as i32 && ny >= 0 && ny < height as i32 {
                let n_pixel = image.get_pixel(nx as u32, ny as u32)[0];
                if n_pixel > 0 {
                    next_dir = Some(dir as u8);
                    break;
                }
            }
        }

        match next_dir {
            Some(dir) => {
                cx += NEIGHBORS[dir as usize].0;
                cy += NEIGHBORS[dir as usize].1;
                came_from = OPPOSITE[dir as usize];
            }
            None => break, // 无路可走
        }
    }

    None
}

/// 提取闭环（没有明显端点的循环）
fn extract_closed_loops(
    image: &GrayImage,
    junction_map: &HashMap<(u32, u32), &Junction>,
    visited: &mut HashSet<((u32, u32), u8)>,
    width: u32,
    height: u32,
) -> Vec<Stroke> {
    let mut strokes = Vec::new();
    let mut visited_pixels = HashSet::new();

    // 收集所有已访问的像素
    for &((x, y), _) in visited.iter() {
        visited_pixels.insert((x, y));
    }

    for y in 0..height {
        for x in 0..width {
            let pixel = image.get_pixel(x, y)[0];
            if pixel == 0 {
                continue;
            }
            if visited_pixels.contains(&(x, y)) {
                continue;
            }
            if junction_map.contains_key(&(x, y)) {
                continue;
            }

            // 发现可能的闭环起点
            if let Some(loop_stroke) = trace_closed_loop(image, (x, y), width, height) {
                strokes.push(loop_stroke);
            }
        }
    }

    strokes
}

/// 追踪单个闭环
fn trace_closed_loop(
    image: &GrayImage,
    start: (u32, u32),
    width: u32,
    height: u32,
) -> Option<Stroke> {
    let mut points = VecDeque::new();
    let mut visited_loop = HashSet::new();
    let mut cx = start.0 as i32;
    let mut cy = start.1 as i32;
    let mut came_from: Option<u8> = None;

    loop {
        let curr = (cx as u32, cy as u32);
        if visited_loop.contains(&curr) {
            if curr == start && points.len() >= 3 {
                // 形成有效闭环
                let subpixel_points: Vec<SubpixelPoint> = points
                    .iter()
                    .map(|&(x, y)| refine_subpixel_centroid(image, x, y))
                    .collect();

                let start_point = subpixel_points[0];
                let end_point = subpixel_points[subpixel_points.len() - 1];
                let mid_points = subpixel_points[1..subpixel_points.len() - 1].to_vec();

                let mut stroke = Stroke {
                    start: start_point,
                    end: end_point,
                    points: mid_points,
                    length: 0.0,
                    is_closed: true,
                };
                stroke.length = stroke.calculate_length();
                return Some(stroke);
            }
            break;
        }

        visited_loop.insert(curr);
        points.push_back(curr);

        // 找下一个方向
        let mut next_dir = None;
        for (dir, &(dx, dy)) in NEIGHBORS.iter().enumerate() {
            if came_from.is_some() && dir as u8 == came_from.unwrap() {
                continue;
            }
            let nx = cx + dx;
            let ny = cy + dy;
            if nx >= 0 && nx < width as i32 && ny >= 0 && ny < height as i32 {
                let n_pixel = image.get_pixel(nx as u32, ny as u32)[0];
                if n_pixel > 0 {
                    next_dir = Some(dir as u8);
                    break;
                }
            }
        }

        match next_dir {
            Some(dir) => {
                cx += NEIGHBORS[dir as usize].0;
                cy += NEIGHBORS[dir as usize].1;
                came_from = Some(OPPOSITE[dir as usize]);
            }
            None => break,
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Luma;

    #[test]
    fn test_detect_endpoint() {
        // 简单直线：x x x
        let mut image = GrayImage::from_pixel(4, 1, Luma([0]));
        image.put_pixel(0, 0, Luma([255]));
        image.put_pixel(1, 0, Luma([255]));
        image.put_pixel(2, 0, Luma([255]));
        image.put_pixel(3, 0, Luma([255]));

        let junctions = detect_junctions(&image);
        // 两端各有 1 个邻居，都是端点
        assert_eq!(
            junctions
                .iter()
                .filter(|j| j.junction_type == JunctionType::Endpoint)
                .count(),
            2
        );
    }

    #[test]
    fn test_detect_l_shape() {
        // L 形拐角应有 2 个不连续方向（更长的臂避免对角邻接问题）
        // x . .
        // x . .
        // x x x
        let mut image = GrayImage::from_pixel(3, 3, Luma([0]));
        image.put_pixel(0, 0, Luma([255]));
        image.put_pixel(0, 1, Luma([255]));
        image.put_pixel(0, 2, Luma([255]));
        image.put_pixel(1, 2, Luma([255]));
        image.put_pixel(2, 2, Luma([255]));

        let junctions = detect_junctions(&image);
        assert!(junctions
            .iter()
            .any(|j| j.junction_type == JunctionType::LShape));
    }

    #[test]
    fn test_detect_t_shape() {
        // T 形
        //   x
        // x x x
        let mut image = GrayImage::from_pixel(4, 3, Luma([0]));
        image.put_pixel(1, 0, Luma([255]));
        image.put_pixel(0, 1, Luma([255]));
        image.put_pixel(1, 1, Luma([255]));
        image.put_pixel(2, 1, Luma([255]));

        let junctions = detect_junctions(&image);
        assert!(junctions
            .iter()
            .any(|j| j.junction_type == JunctionType::TShape));
    }

    #[test]
    fn test_detect_x_shape() {
        // X 形（+）
        //   x
        // x x x
        //   x
        let mut image = GrayImage::from_pixel(3, 3, Luma([0]));
        image.put_pixel(1, 0, Luma([255]));
        image.put_pixel(0, 1, Luma([255]));
        image.put_pixel(1, 1, Luma([255]));
        image.put_pixel(2, 1, Luma([255]));
        image.put_pixel(1, 2, Luma([255]));

        let junctions = detect_junctions(&image);
        assert!(junctions
            .iter()
            .any(|j| j.junction_type == JunctionType::XShape));
    }

    #[test]
    fn test_subpixel_centroid() {
        let mut image = GrayImage::from_pixel(5, 5, Luma([0]));
        // 中心 3x3 亮，右下角更亮（拉动质心）
        image.put_pixel(2, 2, Luma([255]));
        image.put_pixel(3, 2, Luma([128]));
        image.put_pixel(2, 3, Luma([128]));
        image.put_pixel(3, 3, Luma([64]));

        let point = refine_subpixel_centroid(&image, 2, 2);
        assert!(point.x > 2.0);
        assert!(point.y > 2.0);
    }

    #[test]
    fn test_extract_strokes_simple_line() {
        // 简单直线：3 个像素
        let mut image = GrayImage::from_pixel(3, 1, Luma([0]));
        image.put_pixel(0, 0, Luma([255]));
        image.put_pixel(1, 0, Luma([255]));
        image.put_pixel(2, 0, Luma([255]));

        let junctions = detect_junctions(&image);
        let strokes = extract_strokes(&image, &junctions);

        // 两端是端点，应该有一条 stroke
        assert!(!strokes.is_empty());
    }

    #[test]
    fn test_stroke_length() {
        let stroke = Stroke {
            start: SubpixelPoint::new(0.0, 0.0),
            end: SubpixelPoint::new(3.0, 4.0),
            points: vec![],
            length: 0.0,
            is_closed: false,
        };
        assert!((stroke.calculate_length() - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_subpixel_point_distance() {
        let p1 = SubpixelPoint::new(0.0, 0.0);
        let p2 = SubpixelPoint::new(3.0, 4.0);
        assert!((p1.distance(&p2) - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_junction_direction_count() {
        // T 形应该有 3 个方向
        //   x
        // x x x
        let mut image = GrayImage::from_pixel(4, 3, Luma([0]));
        image.put_pixel(1, 0, Luma([255]));
        image.put_pixel(0, 1, Luma([255]));
        image.put_pixel(1, 1, Luma([255]));
        image.put_pixel(2, 1, Luma([255]));

        let junctions = detect_junctions(&image);
        let t_junc = junctions
            .iter()
            .find(|j| j.junction_type == JunctionType::TShape)
            .unwrap();
        assert_eq!(t_junc.directions.len(), 3);
    }
}
