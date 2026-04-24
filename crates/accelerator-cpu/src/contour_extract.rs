//! CPU 轮廓提取实现

use accelerator_api::{AcceleratorError, AcceleratorResult};
use accelerator_api::{ContourExtractConfig, Contours, EdgeMap};

/// CPU 轮廓提取（迭代 DFS）
pub fn extract_contours_cpu(
    edges: &EdgeMap,
    config: &ContourExtractConfig,
) -> AcceleratorResult<Contours> {
    let width = edges.width as usize;
    let height = edges.height as usize;

    if edges.data.len() != width * height {
        return Err(AcceleratorError::InvalidDataFormat(format!(
            "边缘图数据大小不匹配：期望 {}，实际 {}",
            width * height,
            edges.data.len()
        )));
    }

    let mut visited = vec![vec![false; height]; width];
    let mut contours = Vec::new();

    // 8 邻域方向
    let directions: [(isize, isize); 8] = [
        (-1, -1),
        (0, -1),
        (1, -1),
        (-1, 0),
        (1, 0),
        (-1, 1),
        (0, 1),
        (1, 1),
    ];

    for y in 0..height {
        for x in 0..width {
            if !visited[x][y] && edges.data[y * width + x] == 0 {
                let mut contour = Vec::new();

                // 迭代 DFS
                let mut stack = vec![(x, y)];

                while let Some((cx, cy)) = stack.pop() {
                    if cx >= width || cy >= height || visited[cx][cy] {
                        continue;
                    }

                    if edges.data[cy * width + cx] != 0 {
                        continue;
                    }

                    visited[cx][cy] = true;
                    contour.push([cx as f64, cy as f64]);

                    // 添加邻域点
                    for (dx, dy) in &directions {
                        let nx = cx as isize + dx;
                        let ny = cy as isize + dy;

                        if nx >= 0
                            && ny >= 0
                            && (nx as usize) < width
                            && (ny as usize) < height
                            && !visited[nx as usize][ny as usize]
                        {
                            stack.push((nx as usize, ny as usize));
                        }
                    }
                }

                // 简化轮廓
                if contour.len() >= config.min_contour_length {
                    if config.simplify {
                        contour = simplify_douglas_peucker(&contour, config.simplify_epsilon);
                    }

                    if contour.len() >= config.min_contour_length {
                        contours.push(contour);
                    }
                }
            }
        }
    }

    Ok(contours)
}

/// Douglas-Peucker 多边形简化算法（迭代实现）
fn simplify_douglas_peucker(
    points: &[common_types::Point2],
    epsilon: f64,
) -> Vec<common_types::Point2> {
    if points.len() <= 2 {
        return points.to_vec();
    }

    let mut keep = vec![false; points.len()];
    keep[0] = true;
    keep[points.len() - 1] = true;

    let mut stack = vec![(0, points.len() - 1)];

    while let Some((start, end)) = stack.pop() {
        if start + 1 >= end {
            continue;
        }

        let start_point = points[start];
        let end_point = points[end];

        // 找到最远点
        let (max_dist, max_idx) =
            find_farthest_point(&points[start + 1..end], start_point, end_point, start + 1);

        if max_dist > epsilon {
            keep[max_idx] = true;
            stack.push((start, max_idx));
            stack.push((max_idx, end));
        }
    }

    points
        .iter()
        .enumerate()
        .filter(|(i, _)| keep[*i])
        .map(|(_, p)| *p)
        .collect()
}

/// 找到距离线段最远的点
fn find_farthest_point(
    points: &[common_types::Point2],
    start: common_types::Point2,
    end: common_types::Point2,
    offset: usize,
) -> (f64, usize) {
    let mut max_dist = 0.0;
    let mut max_idx = offset;

    for (i, point) in points.iter().enumerate() {
        let dist = point_to_line_distance(*point, start, end);
        if dist > max_dist {
            max_dist = dist;
            max_idx = offset + i;
        }
    }

    (max_dist, max_idx)
}

/// 点到线段的距离
fn point_to_line_distance(
    point: common_types::Point2,
    line_start: common_types::Point2,
    line_end: common_types::Point2,
) -> f64 {
    let dx = line_end[0] - line_start[0];
    let dy = line_end[1] - line_start[1];
    let len_sq = dx * dx + dy * dy;

    if len_sq == 0.0 {
        // 线段退化为点
        return ((point[0] - line_start[0]).powi(2) + (point[1] - line_start[1]).powi(2)).sqrt();
    }

    let t = ((point[0] - line_start[0]) * dx + (point[1] - line_start[1]) * dy) / len_sq;
    let t = t.clamp(0.0, 1.0);

    let proj_x = line_start[0] + t * dx;
    let proj_y = line_start[1] + t * dy;

    ((point[0] - proj_x).powi(2) + (point[1] - proj_y).powi(2)).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_contour_extract_simple() {
        let edges = EdgeMap {
            width: 10,
            height: 10,
            data: {
                let mut data = vec![255u8; 100];
                // 画一个矩形框
                for x in 2..8 {
                    data[2 * 10 + x] = 0;
                    data[7 * 10 + x] = 0;
                }
                for y in 2..8 {
                    data[y * 10 + 2] = 0;
                    data[y * 10 + 7] = 0;
                }
                data
            },
        };

        let config = ContourExtractConfig::default();
        let result = extract_contours_cpu(&edges, &config).unwrap();

        assert!(!result.is_empty());
    }
}
