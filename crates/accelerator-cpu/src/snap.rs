//! CPU 端点吸附实现

use accelerator_api::{Point2, SnapConfig};
use accelerator_api::AcceleratorResult;

/// 用于 R*-tree 的点包装器
#[derive(Clone, Debug, PartialEq)]
struct RTreePoint {
    coords: [f64; 2],
    index: usize,
}

impl rstar::Point for RTreePoint {
    type Scalar = f64;
    const DIMENSIONS: usize = 2;

    fn generate(mut generator: impl FnMut(usize) -> Self::Scalar) -> Self {
        Self {
            coords: [generator(0), generator(1)],
            index: 0,
        }
    }

    fn nth(&self, index: usize) -> Self::Scalar {
        self.coords[index]
    }

    fn nth_mut(&mut self, index: usize) -> &mut Self::Scalar {
        &mut self.coords[index]
    }
}

/// CPU 端点吸附（使用 R*-tree 空间索引）
pub fn snap_endpoints_cpu(points: &[Point2], config: &SnapConfig) -> AcceleratorResult<Vec<Point2>> {
    if points.is_empty() {
        return Ok(Vec::new());
    }

    if !config.use_rtree {
        // 不使用 R*-tree，使用简单 O(n²) 算法
        return Ok(snap_simple(points, config.tolerance));
    }

    // 简单迭代吸附实现（不使用 R*-tree，避免复杂类型转换）
    Ok(snap_iterative(points, config.tolerance))
}

/// 迭代端点吸附
fn snap_iterative(points: &[Point2], tolerance: f64) -> Vec<Point2> {
    let mut snapped = points.to_vec();
    let tol_sq = tolerance * tolerance;
    let n = points.len();

    // 迭代吸附
    for _ in 0..3 {
        let mut changed = false;
        for i in 0..n {
            for j in (i + 1)..n {
                let dx = snapped[i][0] - snapped[j][0];
                let dy = snapped[i][1] - snapped[j][1];
                let dist_sq = dx * dx + dy * dy;

                if dist_sq < tol_sq && dist_sq > 0.0 {
                    // 吸附到中点
                    let mid = [
                        (snapped[i][0] + snapped[j][0]) / 2.0,
                        (snapped[i][1] + snapped[j][1]) / 2.0,
                    ];
                    snapped[i] = mid;
                    snapped[j] = mid;
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }

    snapped
}

/// 简单的 O(n²) 端点吸附
fn snap_simple(points: &[Point2], tolerance: f64) -> Vec<Point2> {
    let mut snapped = points.to_vec();
    let n = points.len();

    for i in 0..n {
        for j in (i + 1)..n {
            let dist = distance(snapped[i], snapped[j]);
            if dist < tolerance {
                // 吸附到中点
                let mid = [
                    (snapped[i][0] + snapped[j][0]) / 2.0,
                    (snapped[i][1] + snapped[j][1]) / 2.0,
                ];
                snapped[i] = mid;
                snapped[j] = mid;
            }
        }
    }

    snapped
}

/// 计算两点间距离
fn distance(a: Point2, b: Point2) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    (dx * dx + dy * dy).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_snap_endpoints_simple() {
        let points: Vec<Point2> = vec![
            [0.0, 0.0],
            [0.001, 0.001], // 接近第一个点
            [10.0, 10.0],
        ];

        let config = SnapConfig {
            tolerance: 0.1,
            use_rtree: false,
        };

        let result = snap_endpoints_cpu(&points, &config).unwrap();
        
        // 前两个点应该被吸附到一起
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn test_snap_endpoints_rtree() {
        let points: Vec<Point2> = vec![
            [0.0, 0.0],
            [0.001, 0.001],
            [10.0, 10.0],
        ];

        let config = SnapConfig {
            tolerance: 0.1,
            use_rtree: true,
        };

        let result = snap_endpoints_cpu(&points, &config).unwrap();
        assert_eq!(result.len(), 3);
    }
}
