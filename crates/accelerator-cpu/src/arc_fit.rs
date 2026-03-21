//! CPU 圆弧拟合实现

use accelerator_api::{Arc as AcceleratorArc, ArcFitConfig, Point2};
use accelerator_api::{AcceleratorResult, AcceleratorError};

/// CPU 圆弧拟合（Kåsa 方法）
pub fn fit_arc_cpu(points: &[Point2], _config: &ArcFitConfig) -> AcceleratorResult<AcceleratorArc> {
    if points.len() < 3 {
        return Err(AcceleratorError::InvalidDataFormat(
            "圆弧拟合至少需要 3 个点".to_string()
        ));
    }

    // Kåsa 圆弧拟合算法
    let n = points.len() as f64;
    
    // 计算质心
    let centroid = [
        points.iter().map(|p| p[0]).sum::<f64>() / n,
        points.iter().map(|p| p[1]).sum::<f64>() / n,
    ];

    // 构建线性方程组
    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    let mut sum_xx = 0.0;
    let mut sum_yy = 0.0;
    let mut sum_xy = 0.0;
    let mut sum_xxx = 0.0;
    let mut sum_yyy = 0.0;
    let mut sum_xyy = 0.0;
    let mut sum_xxy = 0.0;

    for &point in points {
        let x = point[0] - centroid[0];
        let y = point[1] - centroid[1];
        let xx = x * x;
        let yy = y * y;
        let xy = x * y;

        sum_x += x;
        sum_y += y;
        sum_xx += xx;
        sum_yy += yy;
        sum_xy += xy;
        sum_xxx += x * xx;
        sum_yyy += y * yy;
        sum_xyy += x * yy;
        sum_xxy += y * xx;
    }

    // 解线性方程组
    let a = n * sum_xx - sum_x * sum_x;
    let b = n * sum_xy - sum_x * sum_y;
    let c = n * sum_yy - sum_y * sum_y;
    let d = n * sum_xxx + n * sum_xyy - (sum_x * sum_xx + sum_x * sum_yy);
    let e = n * sum_xxy + n * sum_yyy - (sum_y * sum_xx + sum_y * sum_yy);

    let det = a * c - b * b;
    if det.abs() < 1e-10 {
        // 退化为直线，返回一个"无限大"的圆弧
        return Ok(AcceleratorArc::new(centroid, 1e10, 0.0, std::f64::consts::PI * 2.0));
    }

    let center_x = (c * d - b * e) / (2.0 * det);
    let center_y = (a * e - b * d) / (2.0 * det);

    let center = [
        center_x + centroid[0],
        center_y + centroid[1],
    ];

    // 计算半径（平均距离）
    let radius = points
        .iter()
        .map(|p| {
            let dx = p[0] - center[0];
            let dy = p[1] - center[1];
            (dx * dx + dy * dy).sqrt()
        })
        .sum::<f64>()
        / n;

    // 计算起始和终止角度
    let angles: Vec<f64> = points
        .iter()
        .map(|p| (p[1] - center[1]).atan2(p[0] - center[0]))
        .collect();

    let start_angle = angles.iter().cloned().fold(f64::INFINITY, f64::min);
    let end_angle = angles.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    Ok(AcceleratorArc::new(center, radius, start_angle, end_angle))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arc_fit_circle() {
        // 拟合一个圆上的点
        let points: Vec<Point2> = (0..8)
            .map(|i| {
                let angle = i as f64 * std::f64::consts::PI / 4.0;
                [angle.cos() * 10.0, angle.sin() * 10.0]
            })
            .collect();

        let config = ArcFitConfig::default();
        let arc = fit_arc_cpu(&points, &config).unwrap();

        assert!((arc.radius - 10.0).abs() < 1.0);
    }

    #[test]
    fn test_arc_fit_insufficient_points() {
        let points: Vec<Point2> = vec![[0.0, 0.0], [1.0, 1.0]];
        let config = ArcFitConfig::default();
        
        let result = fit_arc_cpu(&points, &config);
        assert!(result.is_err());
    }
}
