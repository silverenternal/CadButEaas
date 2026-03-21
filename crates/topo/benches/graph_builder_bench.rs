//! GraphBuilder 性能基准测试
//!
//! 测试 R*-tree 空间索引的性能改进（O(n²) → O(n log n)）

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use topo::graph_builder::GraphBuilder;
use common_types::LengthUnit;

/// 生成测试用的多段线数据
fn generate_polylines(num_segments: usize, spacing: f64) -> Vec<Vec<[f64; 2]>> {
    (0..num_segments)
        .map(|i| {
            let x = i as f64 * spacing;
            vec![[x, 0.0], [x + spacing * 0.8, 0.0]]
        })
        .collect()
}

/// 生成需要端点吸附的多段线（端点接近但不完全重合）
fn generate_snapping_polylines(num_segments: usize, spacing: f64, tolerance: f64) -> Vec<Vec<[f64; 2]>> {
    let mut polylines = Vec::new();
    for i in 0..num_segments {
        let x = i as f64 * spacing;
        // 每个线段的起点与前一个线段的终点有微小偏移（在容差范围内）
        let start_x = x + tolerance * 0.5;
        polylines.push(vec![[start_x, 0.0], [x + spacing, 0.0]]);
    }
    polylines
}

/// 基准测试：基础图构建（无吸附）
fn bench_graph_builder_basic(c: &mut Criterion) {
    let mut group = c.benchmark_group("graph_builder_basic");

    for &size in &[100, 500, 1000, 5000, 10000] {
        let polylines = generate_polylines(size, 1.0);

        group.bench_with_input(BenchmarkId::from_parameter(size), &polylines, |b, polylines| {
            b.iter(|| {
                let mut builder = GraphBuilder::new(0.5, LengthUnit::Mm);
                builder.snap_and_build(black_box(polylines));
                builder.num_points()
            })
        });
    }

    group.finish();
}

/// 基准测试：端点吸附（R*-tree 优化的主要场景）
fn bench_graph_builder_snapping(c: &mut Criterion) {
    let mut group = c.benchmark_group("graph_builder_snapping");

    for &size in &[100, 500, 1000, 5000, 10000] {
        let polylines = generate_snapping_polylines(size, 1.0, 0.1);

        group.bench_with_input(BenchmarkId::from_parameter(size), &polylines, |b, polylines| {
            b.iter(|| {
                let mut builder = GraphBuilder::new(0.5, LengthUnit::Mm);
                builder.snap_and_build(black_box(polylines));
                builder.num_points()
            })
        });
    }

    group.finish();
}

/// 基准测试：大规模场景（验证 R*-tree 的性能优势）
fn bench_graph_builder_large_scale(c: &mut Criterion) {
    let mut group = c.benchmark_group("graph_builder_large_scale");

    for &size in &[1000, 5000, 10000, 50000, 100000] {
        let polylines = generate_polylines(size, 0.1);

        group.bench_with_input(BenchmarkId::from_parameter(size), &polylines, |b, polylines| {
            b.iter(|| {
                let mut builder = GraphBuilder::new(0.05, LengthUnit::Mm);
                builder.snap_and_build(black_box(polylines));
                builder.num_points()
            })
        });
    }

    group.finish();
}

/// 基准测试：密集场景（大量线段在狭小空间内）
fn bench_graph_builder_dense(c: &mut Criterion) {
    let mut group = c.benchmark_group("graph_builder_dense");

    for &size in &[100, 500, 1000] {
        // 在 10x10 的区域内生成大量线段
        let polylines: Vec<Vec<[f64; 2]>> = (0..size)
            .map(|i| {
                let x = (i as f64 * 0.1) % 10.0;
                let y = (i as f64 * 0.05) % 10.0;
                vec![[x, y], [x + 0.5, y + 0.5]]
            })
            .collect();

        group.bench_with_input(BenchmarkId::from_parameter(size), &polylines, |b, polylines| {
            b.iter(|| {
                let mut builder = GraphBuilder::new(0.5, LengthUnit::Mm);
                builder.snap_and_build(black_box(polylines));
                builder.num_points()
            })
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_graph_builder_basic,
    bench_graph_builder_snapping,
    bench_graph_builder_large_scale,
    bench_graph_builder_dense,
);

criterion_main!(benches);
