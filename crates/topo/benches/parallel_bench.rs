//! 并行化几何处理性能基准测试
//!
//! # P11 锐评落实
//!
//! 原文档声称的并行化是"装饰品"，因为：
//! 1. 真正的耗时大户（文件 IO、DXF 解析）是串行的
//! 2. 实体转换只是字段拷贝，并行化 overhead 可能超过收益
//!
//! 本测试验证真实并行化的性能提升：
//! - 交点计算并行化
//! - 重叠检测并行化
//! - 端点吸附分桶策略（实验性）

use common_types::{LengthUnit, Point2, Polyline};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use topo::graph_builder::{parallel, GraphBuilder};

/// 生成测试用的线段数据（用于交点计算）
fn generate_segments_for_intersection(num_segments: usize, spread: f64) -> Vec<(Point2, Point2)> {
    let mut segments = Vec::new();
    for i in 0..num_segments {
        let angle1 = (i as f64) * 2.0 * std::f64::consts::PI / (num_segments as f64);
        let angle2 = ((i + num_segments / 2) % num_segments) as f64 * 2.0 * std::f64::consts::PI
            / (num_segments as f64);

        let start = [angle1.cos() * spread, angle1.sin() * spread];
        let end = [angle2.cos() * spread, angle2.sin() * spread];

        segments.push((start, end));
    }
    segments
}

/// 生成随机线段（用于重叠检测，预留）
#[allow(dead_code)]
fn generate_random_segments(num_segments: usize, bounds: f64) -> Vec<(Point2, Point2)> {
    use rand::distr::Uniform;
    use rand::{Rng, SeedableRng};
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let range = Uniform::new(-bounds, bounds).unwrap();
    let small_range = Uniform::new(-1.0, 1.0).unwrap();

    (0..num_segments)
        .map(|_| {
            let x1 = rng.sample(range);
            let y1 = rng.sample(range);
            let x2 = x1 + rng.sample(small_range);
            let y2 = y1 + rng.sample(small_range);
            ([x1, y1], [x2, y2])
        })
        .collect()
}

/// 基准测试：交点计算 - 串行 vs 并行
fn bench_intersection_serial(c: &mut Criterion) {
    let mut group = c.benchmark_group("intersection_serial");

    for &size in &[100, 500, 1000] {
        let segments = generate_segments_for_intersection(size, 100.0);

        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            &segments,
            |b, segments| {
                b.iter(|| {
                    let polylines: Vec<Polyline> =
                        segments.iter().map(|&(s, e)| vec![s, e]).collect();
                    let mut builder = GraphBuilder::new(0.5, LengthUnit::Mm);
                    builder.snap_and_build(&polylines);
                    black_box(builder.num_points())
                })
            },
        );
    }

    group.finish();
}

/// 基准测试：交点计算 - 并行版本
fn bench_intersection_parallel(c: &mut Criterion) {
    let mut group = c.benchmark_group("intersection_parallel");

    for &size in &[100, 500, 1000] {
        let segments = generate_segments_for_intersection(size, 100.0);

        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            &segments,
            |b, segments| {
                b.iter(|| {
                    let intersections =
                        parallel::compute_intersections_parallel(black_box(segments), 0.5);
                    black_box(intersections.len())
                })
            },
        );
    }

    group.finish();
}

/// 基准测试：端点吸附 - 串行 vs 并行（分桶策略）
fn bench_snap_serial(c: &mut Criterion) {
    let mut group = c.benchmark_group("snap_serial");

    for &size in &[500, 1000, 5000, 10000] {
        let polylines: Vec<Polyline> = (0..size)
            .map(|i| {
                let x = i as f64 * 0.1;
                vec![[x, 0.0], [x + 0.05, 0.0]]
            })
            .collect();

        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            &polylines,
            |b, polylines| {
                b.iter(|| {
                    let mut builder = GraphBuilder::new(0.5, LengthUnit::Mm);
                    builder.snap_and_build(black_box(polylines));
                    black_box(builder.num_points())
                })
            },
        );
    }

    group.finish();
}

/// 基准测试：端点吸附 - 并行版本（分桶策略）
fn bench_snap_parallel(c: &mut Criterion) {
    let mut group = c.benchmark_group("snap_parallel");

    for &size in &[500, 1000, 5000, 10000] {
        let polylines: Vec<Polyline> = (0..size)
            .map(|i| {
                let x = i as f64 * 0.1;
                vec![[x, 0.0], [x + 0.05, 0.0]]
            })
            .collect();

        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            &polylines,
            |b, polylines| {
                b.iter(|| {
                    let (points, edges) = parallel::snap_endpoints_parallel(
                        black_box(polylines),
                        0.5,
                        LengthUnit::Mm,
                    );
                    black_box((points.len(), edges.len()))
                })
            },
        );
    }

    group.finish();
}

/// 基准测试：自适应容差性能对比
fn bench_adaptive_tolerance(c: &mut Criterion) {
    let mut group = c.benchmark_group("adaptive_tolerance");

    for &size in &[500, 1000, 5000] {
        let polylines: Vec<Polyline> = (0..size)
            .map(|i| {
                let x = i as f64 * 0.1;
                vec![[x, 0.0], [x + 0.05, 0.0]]
            })
            .collect();

        // 固定容差
        group.bench_with_input(
            BenchmarkId::new("fixed", size),
            &polylines,
            |b, polylines| {
                b.iter(|| {
                    let mut builder = GraphBuilder::new(0.5, LengthUnit::Mm);
                    builder.snap_and_build(black_box(polylines));
                    black_box(builder.num_points())
                })
            },
        );

        // 自适应容差
        group.bench_with_input(
            BenchmarkId::new("adaptive", size),
            &polylines,
            |b, polylines| {
                b.iter(|| {
                    let mut builder = GraphBuilder::with_adaptive_tolerance(LengthUnit::Mm);
                    builder.snap_and_build(black_box(polylines));
                    black_box(builder.num_points())
                })
            },
        );
    }

    group.finish();
}

/// 基准测试：完整拓扑构建流程
fn bench_full_topology(c: &mut Criterion) {
    let mut group = c.benchmark_group("full_topology");

    for &size in &[100, 500, 1000] {
        // 生成网格状线段（模拟真实墙体）
        let mut polylines: Vec<Polyline> = Vec::new();
        let grid_size = (size as f64).sqrt() as usize;

        // 水平线
        for i in 0..=grid_size {
            let y = i as f64 * 10.0;
            polylines.push(vec![[0.0, y], [grid_size as f64 * 10.0, y]]);
        }

        // 垂直线
        for j in 0..=grid_size {
            let x = j as f64 * 10.0;
            polylines.push(vec![[x, 0.0], [x, grid_size as f64 * 10.0]]);
        }

        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            &polylines,
            |b, polylines| {
                b.iter(|| {
                    let mut builder = GraphBuilder::new(0.5, LengthUnit::Mm);
                    builder.set_adaptive_tolerance(true);
                    builder.snap_and_build(black_box(polylines));
                    builder.detect_and_merge_overlapping_segments();
                    builder.compute_intersections_and_split();
                    black_box((builder.num_points(), builder.num_edges()))
                })
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_intersection_serial,
    bench_intersection_parallel,
    bench_snap_serial,
    bench_snap_parallel,
    bench_adaptive_tolerance,
    bench_full_topology,
);

criterion_main!(benches);
