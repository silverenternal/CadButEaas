//! P11-6 性能基准测试矩阵
//!
//! ## 测试目标
//!
//! 量化 P11 阶段各项优化的性能提升：
//! 1. Bentley-Ottmann vs R*-tree 交点检测
//! 2. UnionFind 并行化性能
//! 3. GPU vs egui 渲染性能（预留）
//! 4. Halfedge 遍历性能（预留）
//!
//! ## 运行方法
//!
//! ```bash
//! # 运行所有基准测试
//! cargo bench --package topo --bench benchmark_suite
//!
//! # 运行特定测试
//! cargo bench --package topo --bench benchmark_suite -- bentley_ottmann
//! ```

use common_types::Point2;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::time::Duration;
use topo::bentley_ottmann::BentleyOttmann;
use topo::graph_builder::GraphBuilder;
use topo::union_find::UnionFind;

/// 生成随机线段（使用固定种子以保证可重复性）
fn generate_random_segments(n: usize, range: f64) -> Vec<(Point2, Point2)> {
    let mut rng = ChaCha8Rng::seed_from_u64(42); // 固定种子
    (0..n)
        .map(|_| {
            let start = [
                rng.random_range(-range..range),
                rng.random_range(-range..range),
            ];
            let end = [
                rng.random_range(-range..range),
                rng.random_range(-range..range),
            ];
            (start, end)
        })
        .collect()
}

/// 生成网格状线段（密集交叉场景）
fn generate_grid_segments(rows: usize, cols: usize, spacing: f64) -> Vec<(Point2, Point2)> {
    let mut segments = Vec::new();

    // 水平线段
    for i in 0..rows {
        let y = i as f64 * spacing;
        for j in 0..cols {
            let x1 = j as f64 * spacing;
            let x2 = x1 + spacing;
            segments.push(([x1, y], [x2, y]));
        }
    }

    // 垂直线段
    for j in 0..cols {
        let x = j as f64 * spacing;
        for i in 0..rows {
            let y1 = i as f64 * spacing;
            let y2 = y1 + spacing;
            segments.push(([x, y1], [x, y2]));
        }
    }

    segments
}

/// 生成用于并查集测试的 union 对
fn generate_union_pairs(n: usize) -> Vec<(usize, usize)> {
    (0..n / 2).map(|i| (i * 2, i * 2 + 1)).collect()
}

// ============================================================================
// Bentley-Ottmann vs R*-tree 基准测试
// ============================================================================

fn bench_bentley_ottmann_small(c: &mut Criterion) {
    let segments = generate_random_segments(100, 1000.0);
    let bo_segments: Vec<topo::bentley_ottmann::Segment> = segments
        .iter()
        .map(|&(s, e)| topo::bentley_ottmann::Segment::new(s, e))
        .collect();

    c.bench_function("bentley_ottmann/100_segments", |b| {
        b.iter(|| {
            let mut bo = BentleyOttmann::new();
            bo.find_intersections(black_box(&bo_segments))
        })
    });
}

fn bench_bentley_ottmann_medium(c: &mut Criterion) {
    let segments = generate_random_segments(500, 1000.0);
    let bo_segments: Vec<topo::bentley_ottmann::Segment> = segments
        .iter()
        .map(|&(s, e)| topo::bentley_ottmann::Segment::new(s, e))
        .collect();

    c.bench_function("bentley_ottmann/500_segments", |b| {
        b.iter(|| {
            let mut bo = BentleyOttmann::new();
            bo.find_intersections(black_box(&bo_segments))
        })
    });
}

fn bench_bentley_ottmann_large(c: &mut Criterion) {
    let segments = generate_random_segments(1000, 1000.0);
    let bo_segments: Vec<topo::bentley_ottmann::Segment> = segments
        .iter()
        .map(|&(s, e)| topo::bentley_ottmann::Segment::new(s, e))
        .collect();

    c.bench_function("bentley_ottmann/1000_segments", |b| {
        b.iter(|| {
            let mut bo = BentleyOttmann::new();
            bo.find_intersections(black_box(&bo_segments))
        })
    });
}

fn bench_bentley_ottmann_grid(c: &mut Criterion) {
    // 10x10 网格 = 200 线段，100 交点
    let segments = generate_grid_segments(10, 10, 100.0);
    let bo_segments: Vec<topo::bentley_ottmann::Segment> = segments
        .iter()
        .map(|&(s, e)| topo::bentley_ottmann::Segment::new(s, e))
        .collect();

    c.bench_function("bentley_ottmann/10x10_grid", |b| {
        b.iter(|| {
            let mut bo = BentleyOttmann::new();
            bo.find_intersections(black_box(&bo_segments))
        })
    });
}

fn bench_graph_builder_rtree(c: &mut Criterion) {
    let segments = generate_random_segments(500, 1000.0);

    c.bench_function("graph_builder_rtree/500_segments", |b| {
        b.iter(|| {
            let mut builder = GraphBuilder::new(0.5, common_types::LengthUnit::Mm);
            let polylines: Vec<Vec<Point2>> = segments.iter().map(|&(s, e)| vec![s, e]).collect();
            builder.snap_and_build(&polylines);
            builder.compute_intersections_and_split();
        })
    });
}

fn bench_graph_builder_bentley_ottmann(c: &mut Criterion) {
    let segments = generate_random_segments(500, 1000.0);

    c.bench_function("graph_builder_bentley_ottmann/500_segments", |b| {
        b.iter(|| {
            let mut builder = GraphBuilder::new(0.5, common_types::LengthUnit::Mm);
            let polylines: Vec<Vec<Point2>> = segments.iter().map(|&(s, e)| vec![s, e]).collect();
            builder.snap_and_build(&polylines);
            builder.compute_intersections_bentley_ottmann();
        })
    });
}

// ============================================================================
// UnionFind 性能基准测试
// ============================================================================

fn bench_union_find_small(c: &mut Criterion) {
    let n = 1000;
    let unions = generate_union_pairs(n);

    c.bench_function("union_find/1000_elements", |b| {
        b.iter(|| {
            let mut uf = UnionFind::new(n);
            uf.union_parallel(black_box(&unions));
            uf.component_count()
        })
    });
}

fn bench_union_find_medium(c: &mut Criterion) {
    let n = 10000;
    let unions = generate_union_pairs(n);

    c.bench_function("union_find/10000_elements", |b| {
        b.iter(|| {
            let mut uf = UnionFind::new(n);
            uf.union_parallel(black_box(&unions));
            uf.component_count()
        })
    });
}

fn bench_union_find_large(c: &mut Criterion) {
    let n = 100000;
    let unions = generate_union_pairs(n);

    c.bench_function("union_find/100000_elements", |b| {
        b.iter(|| {
            let mut uf = UnionFind::new(n);
            uf.union_parallel(black_box(&unions));
            uf.component_count()
        })
    });
}

// ============================================================================
// 对比基准测试：Bentley-Ottmann vs R*-tree
// ============================================================================

fn bench_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("bentley_ottmann_vs_rtree");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(30));

    for size in [100, 500, 1000, 2000].iter() {
        let segments = generate_random_segments(*size, 1000.0);
        let polylines: Vec<Vec<Point2>> = segments.iter().map(|&(s, e)| vec![s, e]).collect();

        // R*-tree 方法
        group.bench_with_input(
            BenchmarkId::new("rtree", size),
            &polylines,
            |b, polylines| {
                b.iter(|| {
                    let mut builder = GraphBuilder::new(0.5, common_types::LengthUnit::Mm);
                    builder.snap_and_build(polylines);
                    builder.compute_intersections_and_split();
                })
            },
        );

        // Bentley-Ottmann 方法
        group.bench_with_input(
            BenchmarkId::new("bentley_ottmann", size),
            &polylines,
            |b, polylines| {
                b.iter(|| {
                    let mut builder = GraphBuilder::new(0.5, common_types::LengthUnit::Mm);
                    builder.snap_and_build(polylines);
                    builder.compute_intersections_bentley_ottmann();
                })
            },
        );
    }

    group.finish();
}

// ============================================================================
// 网格场景基准测试（密集交叉）
// ============================================================================

fn bench_grid_scenarios(c: &mut Criterion) {
    let mut group = c.benchmark_group("grid_scenarios");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(30));

    for &(rows, cols) in &[(5, 5), (10, 10), (20, 20)] {
        let segments = generate_grid_segments(rows, cols, 100.0);
        let polylines: Vec<Vec<Point2>> = segments.iter().map(|&(s, e)| vec![s, e]).collect();

        let name = format!("{}x{}_grid", rows, cols);

        // R*-tree 方法
        group.bench_with_input(
            BenchmarkId::new("rtree", &name),
            &polylines,
            |b, polylines| {
                b.iter(|| {
                    let mut builder = GraphBuilder::new(0.5, common_types::LengthUnit::Mm);
                    builder.snap_and_build(polylines);
                    builder.compute_intersections_and_split();
                })
            },
        );

        // Bentley-Ottmann 方法
        group.bench_with_input(
            BenchmarkId::new("bentley_ottmann", &name),
            &polylines,
            |b, polylines| {
                b.iter(|| {
                    let mut builder = GraphBuilder::new(0.5, common_types::LengthUnit::Mm);
                    builder.snap_and_build(polylines);
                    builder.compute_intersections_bentley_ottmann();
                })
            },
        );
    }

    group.finish();
}

// ============================================================================
// 配置和入口
// ============================================================================

criterion_group!(
    name = benches;
    config = Criterion::default()
        .sample_size(20)
        .measurement_time(Duration::from_secs(10))
        .warm_up_time(Duration::from_secs(2))
        .noise_threshold(0.05);
    targets =
        // Bentley-Ottmann 单测
        bench_bentley_ottmann_small,
        bench_bentley_ottmann_medium,
        bench_bentley_ottmann_large,
        bench_bentley_ottmann_grid,
        // GraphBuilder 对比
        bench_graph_builder_rtree,
        bench_graph_builder_bentley_ottmann,
        // UnionFind 性能
        bench_union_find_small,
        bench_union_find_medium,
        bench_union_find_large,
        // 对比测试
        bench_comparison,
        bench_grid_scenarios,
        // P0-5 新增：大规模性能基准测试矩阵
        bench_large_scale_comparison,
);

criterion_main!(benches);

// ============================================================================
// P0-5 新增：大规模性能基准测试矩阵（锐评落实）
// ============================================================================

/// 大规模场景对比测试（10000/50000/100000 线段）
fn bench_large_scale_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("large_scale_comparison");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(60)); // 延长测试时间

    for &size in &[10000, 50000, 100000] {
        let segments = generate_random_segments(size, 10000.0);
        let polylines: Vec<Vec<Point2>> = segments.iter().map(|&(s, e)| vec![s, e]).collect();

        // R*-tree 方法
        group.bench_with_input(
            BenchmarkId::new("rtree", size),
            &polylines,
            |b, polylines| {
                b.iter(|| {
                    let mut builder = GraphBuilder::new(0.5, common_types::LengthUnit::Mm);
                    builder.snap_and_build(polylines);
                    builder.compute_intersections_and_split();
                })
            },
        );

        // Bentley-Ottmann 方法（O((n+k) log n) 复杂度）
        group.bench_with_input(
            BenchmarkId::new("bentley_ottmann", size),
            &polylines,
            |b, polylines| {
                b.iter(|| {
                    let mut builder = GraphBuilder::new(0.5, common_types::LengthUnit::Mm);
                    builder.snap_and_build(polylines);
                    builder.compute_intersections_bentley_ottmann();
                })
            },
        );
    }

    group.finish();
}
