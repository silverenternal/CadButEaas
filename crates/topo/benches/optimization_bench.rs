//! CAD 几何处理系统 - 算法性能优化基准测试
//!
//! # 测试目标
//!
//! 验证 P11 级别性能优化的效果：
//! - P0: Bentley-Ottmann 扫描线算法（10-100x 提升）
//! - P0: 并行端点吸附（3-5x 提升）
//! - P1: NURBS 解析导数（3x 提升）
//! - P2: 圆弧拟合预过滤（3x 提升）
//!
//! # 使用方法
//!
//! ```bash
//! # 运行所有基准测试
//! cargo bench --bench optimization_bench
//!
//! # 运行特定测试
//! cargo bench --bench optimization_bench -- --test benched
//! ```

use common_types::{LengthUnit, Point2, Polyline};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

// ============================================================================
// 测试数据生成
// ============================================================================

/// 生成测试用的多段线数据（线性排列，预留）
#[allow(dead_code)]
fn generate_linear_polylines(num_segments: usize, spacing: f64) -> Vec<Polyline> {
    (0..num_segments)
        .map(|i| {
            let x = i as f64 * spacing;
            vec![[x, 0.0], [x + spacing * 0.8, 0.0]]
        })
        .collect()
}

/// 生成网格状多段线（用于交点检测测试）
fn generate_grid_polylines(grid_size: usize, spacing: f64) -> Vec<Polyline> {
    let mut polylines = Vec::new();

    // 水平线
    for i in 0..=grid_size {
        let y = i as f64 * spacing;
        polylines.push(vec![[0.0, y], [grid_size as f64 * spacing, y]]);
    }

    // 垂直线
    for i in 0..=grid_size {
        let x = i as f64 * spacing;
        polylines.push(vec![[x, 0.0], [x, grid_size as f64 * spacing]]);
    }

    polylines
}

/// 生成需要端点吸附的多段线
fn generate_snapping_polylines(num_segments: usize, spacing: f64, tolerance: f64) -> Vec<Polyline> {
    let mut polylines = Vec::new();
    for i in 0..num_segments {
        let x = i as f64 * spacing;
        let start_x = x + tolerance * 0.5;
        polylines.push(vec![[start_x, 0.0], [x + spacing, 0.0]]);
    }
    polylines
}

/// 生成圆弧测试数据
fn generate_arc_polylines(num_arcs: usize, radius: f64) -> Vec<Polyline> {
    use std::f64::consts::PI;

    (0..num_arcs)
        .map(|i| {
            let angle_step = PI / 18.0; // 10 段圆弧
            let start_angle = (i as f64) * 0.1;
            (0..=10)
                .map(|j| {
                    let angle = start_angle + j as f64 * angle_step;
                    [radius * angle.cos(), radius * angle.sin()]
                })
                .collect()
        })
        .collect()
}

// ============================================================================
// P0: Bentley-Ottmann 扫描线算法基准测试
// ============================================================================

mod bentley_ottmann_bench {
    use super::*;
    use topo::graph_builder::GraphBuilder;

    /// 基准测试：交点检测（小规模场景）
    pub fn bench_intersections_small(c: &mut Criterion) {
        let mut group = c.benchmark_group("intersections_bentley_ottmann");

        for &size in &[50, 100, 200] {
            let polylines = generate_grid_polylines(size, 1.0);

            group.bench_with_input(
                BenchmarkId::from_parameter(format!("grid_{}x{}", size, size)),
                &polylines,
                |b, polylines| {
                    b.iter(|| {
                        let mut builder = GraphBuilder::new(0.5, LengthUnit::Mm);
                        builder.snap_and_build(black_box(polylines));
                        builder.compute_intersections_and_split();
                        builder.num_points()
                    })
                },
            );
        }

        group.finish();
    }

    /// 基准测试：交点检测（大规模场景，使用 Bentley-Ottmann）
    pub fn bench_intersections_large(c: &mut Criterion) {
        let mut group = c.benchmark_group("intersections_bentley_ottmann_large");

        for &size in &[500, 1000, 2000] {
            let polylines = generate_grid_polylines(size, 0.5);

            group.bench_with_input(
                BenchmarkId::from_parameter(format!("grid_{}x{}", size, size)),
                &polylines,
                |b, polylines| {
                    b.iter(|| {
                        let mut builder = GraphBuilder::new(0.5, LengthUnit::Mm);
                        builder.snap_and_build(black_box(polylines));
                        builder.compute_intersections_and_split();
                        builder.num_points()
                    })
                },
            );
        }

        group.finish();
    }
}

// ============================================================================
// P0: 并行端点吸附基准测试
// ============================================================================

mod parallel_snap_bench {
    use super::*;
    use topo::parallel::snap_endpoints_parallel;

    /// 基准测试：并行端点吸附 vs 串行端点吸附
    pub fn bench_parallel_vs_serial(c: &mut Criterion) {
        let mut group = c.benchmark_group("snap_endpoints_parallel_vs_serial");

        for &size in &[100, 500, 1000, 5000, 10000] {
            let points: Vec<Point2> = generate_snapping_polylines(size, 1.0, 0.1)
                .into_iter()
                .flatten()
                .collect();

            group.bench_with_input(BenchmarkId::from_parameter(size), &points, |b, points| {
                b.iter(|| snap_endpoints_parallel(black_box(points), 0.5))
            });
        }

        group.finish();
    }

    /// 基准测试：不同容差下的并行吸附性能
    pub fn bench_parallel_different_tolerance(c: &mut Criterion) {
        let mut group = c.benchmark_group("snap_endpoints_tolerance");

        let points: Vec<Point2> = generate_snapping_polylines(1000, 1.0, 0.1)
            .into_iter()
            .flatten()
            .collect();

        for &tolerance in &[0.1, 0.5, 1.0, 2.0] {
            group.bench_with_input(
                BenchmarkId::from_parameter(format!("tol_{}", tolerance)),
                &tolerance,
                |b, &tol| b.iter(|| snap_endpoints_parallel(black_box(&points), tol)),
            );
        }

        group.finish();
    }
}

// ============================================================================
// P2: 圆弧拟合预过滤基准测试
// 注：圆弧拟合功能已迁移至 vectorize crate，相关基准测试移至 vectorize/benches/
// ============================================================================

// ============================================================================
// 综合性能基准测试
// ============================================================================

mod integrated_bench {
    use super::*;
    use topo::graph_builder::GraphBuilder;

    /// 基准测试：完整流程（典型建筑图纸场景）
    pub fn bench_full_pipeline(c: &mut Criterion) {
        let mut group = c.benchmark_group("full_pipeline_typical");

        for &size in &[500, 1000, 2000] {
            // 生成类似建筑图纸的混合数据
            let mut polylines = generate_grid_polylines((size as f64).sqrt() as usize, 10.0);
            polylines.extend(generate_arc_polylines(size / 10, 5.0));

            group.bench_with_input(
                BenchmarkId::from_parameter(size),
                &polylines,
                |b, polylines| {
                    b.iter(|| {
                        let mut builder = GraphBuilder::new(0.5, LengthUnit::Mm);
                        builder.snap_and_build(black_box(polylines));
                        builder.detect_and_merge_overlapping_segments();
                        builder.compute_intersections_and_split();
                        (builder.num_points(), builder.num_edges())
                    })
                },
            );
        }

        group.finish();
    }
}

// ============================================================================
// Criterion 配置
// ============================================================================

criterion_group!(
    bentley_ottmann_group,
    bentley_ottmann_bench::bench_intersections_small,
    bentley_ottmann_bench::bench_intersections_large,
);

criterion_group!(
    parallel_snap_group,
    parallel_snap_bench::bench_parallel_vs_serial,
    parallel_snap_bench::bench_parallel_different_tolerance,
);

criterion_group!(integrated_group, integrated_bench::bench_full_pipeline,);

criterion_main!(bentley_ottmann_group, parallel_snap_group, integrated_group,);
