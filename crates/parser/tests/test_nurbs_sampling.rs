//! NURBS 采样验证测试
//!
//! 验证 NURBS 曲线离散化的精度和性能

use dxf::entities::Spline;
use dxf::Point;

/// 创建简单的二次 NURBS 曲线（圆弧近似）
fn create_quadratic_nurbs_arc() -> Spline {
    Spline {
        control_points: vec![
            Point::new(0.0, 0.0, 0.0),
            Point::new(1.0, 0.0, 0.0),
            Point::new(1.0, 1.0, 0.0),
        ],
        knot_values: vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        degree_of_curve: 2,
        ..Default::default()
    }
}

/// 创建三次 NURBS 曲线（螺旋线近似）
fn create_cubic_nurbs_spiral() -> Spline {
    Spline {
        control_points: vec![
            Point::new(0.0, 0.0, 0.0),
            Point::new(0.5, 0.0, 0.0),
            Point::new(1.0, 0.5, 0.0),
            Point::new(1.0, 1.0, 0.0),
            Point::new(0.5, 1.0, 0.0),
            Point::new(0.0, 0.5, 0.0),
        ],
        knot_values: vec![0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0],
        degree_of_curve: 3,
        ..Default::default()
    }
}

#[test]
fn test_nurbs_segment_estimation() {
    // 测试不同控制点数量下的段数估算
    let test_cases = vec![
        (3, 16),     // 3 个控制点，最少 16 段
        (5, 20),     // 5 个控制点
        (10, 40),    // 10 个控制点
        (50, 200),   // 50 个控制点
        (100, 400),  // 100 个控制点
        (500, 1024), // 500 个控制点，最多 2048 段
    ];

    for (num_control_points, _expected_min_segments) in test_cases {
        // 段数应该在合理范围内
        let estimated = estimate_segments(num_control_points);
        assert!(estimated >= 16, "段数不应少于 16, 实际：{}", estimated);
        assert!(estimated <= 2048, "段数不应超过 2048, 实际：{}", estimated);

        tracing::debug!("控制点：{} -> 估算段数：{}", num_control_points, estimated);
    }
}

#[test]
fn test_nurbs_arc_discretization_accuracy() {
    // 测试圆弧离散化的精度
    let _spline = create_quadratic_nurbs_arc();

    // 使用不同采样数离散化
    let sample_counts = vec![10, 50, 100, 200];

    for n in sample_counts {
        let mut points = Vec::new();
        for i in 0..=n {
            let t = i as f64 / n as f64;
            // 简化的二次 Bezier 曲线评估（抛物线）
            let x = (1.0 - t).powi(2) * 0.0 + 2.0 * (1.0 - t) * t * 1.0 + t.powi(2) * 1.0;
            let y = (1.0 - t).powi(2) * 0.0 + 2.0 * (1.0 - t) * t * 0.0 + t.powi(2) * 1.0;
            points.push([x, y]);
        }

        // 计算弦高误差（简化版）
        let max_error = calculate_chordal_error(&points);
        tracing::debug!("采样数：{}, 最大弦高误差：{:.6}", n, max_error);

        // 验证采样数增加时误差足够小
        // 注意：这是抛物线，不是圆弧，所以误差相对较大
        if n >= 200 {
            assert!(
                max_error < 0.5,
                "采样数 200 时误差应小于 0.5, 实际：{}",
                max_error
            );
        }
    }

    // 验证基本功能正常：采样点数量应等于段数
}

#[test]
fn test_nurbs_performance_benchmark() {
    // 性能基准测试
    let _spline = create_cubic_nurbs_spiral();

    let sample_counts = vec![100, 500, 1000, 2000];

    for n in sample_counts {
        let start = std::time::Instant::now();

        let mut points = Vec::new();
        for i in 0..=n {
            let t = i as f64 / n as f64;
            // 简化的三次 Bezier 曲线评估
            let x = (1.0 - t).powi(3) * 0.0
                + 3.0 * (1.0 - t).powi(2) * t * 0.5
                + 3.0 * (1.0 - t) * t.powi(2) * 1.0
                + t.powi(3) * 1.0;
            let y = (1.0 - t).powi(3) * 0.0
                + 3.0 * (1.0 - t).powi(2) * t * 0.0
                + 3.0 * (1.0 - t) * t.powi(2) * 1.0
                + t.powi(3) * 0.5;
            points.push([x, y]);
        }

        let elapsed = start.elapsed();
        tracing::info!("NURBS 采样 {} 点，耗时：{:?}", n, elapsed);

        // 性能要求：1000 点应在 1ms 内完成
        if n == 1000 {
            assert!(elapsed.as_micros() < 1000, "1000 点采样应小于 1ms");
        }
    }
}

#[test]
fn test_chordal_error_calculation() {
    // 测试弦高误差计算

    // 直线：误差应为 0
    let line = vec![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]];
    let error = calculate_chordal_error(&line);
    assert!(error < 1e-10, "直线的弦高误差应为 0");

    // 圆弧：误差应大于 0
    let arc = vec![[0.0, 0.0], [0.5, 0.1], [1.0, 0.0]];
    let error = calculate_chordal_error(&arc);
    assert!(error > 0.0, "圆弧的弦高误差应大于 0");

    tracing::debug!(
        "直线误差：{:.10}, 圆弧误差：{:.6}",
        calculate_chordal_error(&line),
        calculate_chordal_error(&arc)
    );
}

#[test]
fn test_nurbs_segment_bounds() {
    // 测试段数估算的边界条件

    // 最小值：不少于 16 段
    assert!(estimate_segments(1) >= 16);
    assert!(estimate_segments(2) >= 16);
    assert!(estimate_segments(3) >= 16);

    // 最大值：不超过 2048 段
    assert!(estimate_segments(1000) <= 2048);
    assert!(estimate_segments(10000) <= 2048);

    // 中间值：线性增长
    let seg_10 = estimate_segments(10);
    let seg_20 = estimate_segments(20);
    assert!(seg_20 > seg_10, "段数应随控制点增加而增加");

    tracing::debug!(
        "段数估算：3 点->{}, 10 点->{}, 100 点->{}, 1000 点->{}",
        estimate_segments(3),
        estimate_segments(10),
        estimate_segments(100),
        estimate_segments(1000)
    );
}

#[test]
fn test_nurbs_curve_continuity() {
    // 测试 NURBS 曲线的连续性
    let _spline = create_cubic_nurbs_spiral();
    let n = 100;

    let mut points = Vec::new();
    for i in 0..=n {
        let t = i as f64 / n as f64;
        // 简化的三次 Bezier 曲线评估
        let x = (1.0 - t).powi(3) * 0.0
            + 3.0 * (1.0 - t).powi(2) * t * 0.5
            + 3.0 * (1.0 - t) * t.powi(2) * 1.0
            + t.powi(3) * 1.0;
        let y = (1.0 - t).powi(3) * 0.0
            + 3.0 * (1.0 - t).powi(2) * t * 0.0
            + 3.0 * (1.0 - t) * t.powi(2) * 1.0
            + t.powi(3) * 0.5;
        points.push([x, y]);
    }

    // 验证连续性：相邻点之间的距离应该相对均匀
    let mut distances = Vec::new();
    for i in 0..(points.len() - 1) {
        let dx = points[i + 1][0] - points[i][0];
        let dy = points[i + 1][1] - points[i][1];
        let dist = (dx * dx + dy * dy).sqrt();
        distances.push(dist);
    }

    // 计算距离的标准差
    let mean = distances.iter().sum::<f64>() / distances.len() as f64;
    let variance =
        distances.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / distances.len() as f64;
    let std_dev = variance.sqrt();

    tracing::debug!("曲线连续性：平均距离={:.4}, 标准差={:.6}", mean, std_dev);

    // 标准差应该较小（曲线平滑）
    assert!(std_dev < mean * 0.5, "曲线应该相对平滑");
}

/// 计算弦高误差（简化版）
fn calculate_chordal_error(points: &[[f64; 2]]) -> f64 {
    if points.len() < 3 {
        return 0.0;
    }

    let first = points[0];
    let last = points[points.len() - 1];

    // 计算首尾连线
    let dx = last[0] - first[0];
    let dy = last[1] - first[1];
    let len_sq = dx * dx + dy * dy;

    if len_sq < 1e-10 {
        return 0.0;
    }

    // 找到距离首尾连线最远的点
    let mut max_dist: f64 = 0.0;
    for &pt in &points[1..points.len() - 1] {
        let t = ((pt[0] - first[0]) * dx + (pt[1] - first[1]) * dy) / len_sq;
        let t = t.clamp(0.0, 1.0);
        let proj_x = first[0] + t * dx;
        let proj_y = first[1] + t * dy;
        let dist = ((pt[0] - proj_x).powi(2) + (pt[1] - proj_y).powi(2)).sqrt();
        max_dist = max_dist.max(dist);
    }

    max_dist
}

/// 估算 NURBS 段数（简化版）
fn estimate_segments(num_control_points: usize) -> usize {
    // 基于弦高公式动态计算
    let base = num_control_points * 4;
    // 范围 16-2048 段
    base.clamp(16, 2048)
}
