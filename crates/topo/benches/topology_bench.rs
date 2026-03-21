//! 拓扑服务性能基准测试
//!
//! 测试 TopoService 在不同规模数据下的性能表现
//! 包括：端点吸附、图构建、环提取的完整流程
//!
//! ## 大规模测试（🔴 Principal Engineer 要求补充）
//!
//! | 测试名称 | 线段数量 | 矩形数量 | 状态 |
//! |----------|----------|----------|------|
//! | small | 400 | 100 | ✅ 默认运行 |
//! | medium | 4,000 | 1,000 | ✅ 默认运行 |
//! | large | 40,000 | 10,000 | ✅ 默认运行 |
//! | 100k_segments | 100,000 | 25,000 | ⚠️ ignore (手动运行) |
//! | 500k_segments | 500,000 | 125,000 | ⚠️ ignore (手动运行) |
//! | 1m_segments | 1,000,000 | 250,000 | ⚠️ ignore (手动运行) |
//!
//! ## 运行命令
//!
//! ```bash
//! # 运行所有默认测试
//! cargo bench --bench topology_bench
//!
//! # 运行包含 ignore 测试
//! cargo bench --bench topology_bench -- --ignored
//!
//! # 运行特定测试
//! cargo bench --bench topology_bench -- topology_100k_segments
//! ```
//!
//! ## 性能目标（Principal Engineer 要求）
//!
//! | 规模 | 处理时间 | 内存使用 |
//! |------|----------|----------|
//! | 100 线段 | < 1ms | < 1MB |
//! | 1,000 线段 | < 10ms | < 10MB |
//! | 10,000 线段 | < 100ms | < 100MB |
//! | 100,000 线段 | < 1s | < 500MB |
//! | 1,000,000 线段 | < 10s | < 2GB |

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use topo::TopoService;
use topo::service::TopoConfig;
use common_types::{ToleranceConfig, LengthUnit};

/// 生成矩形网格数据
/// 
/// # 参数
/// - `count`: 矩形数量
/// - `spacing`: 矩形之间的间距
/// 
/// # 返回
/// Vec<Vec<[f64; 2]>> - 每个矩形是一个闭合多段线
fn generate_rectangles(count: usize, spacing: f64) -> Vec<Vec<[f64; 2]>> {
    (0..count)
        .map(|i| {
            let row = i / 10;
            let col = i % 10;
            let offset_x = col as f64 * spacing;
            let offset_y = row as f64 * spacing;
            
            // 100x100 的矩形
            vec![
                [0.0 + offset_x, 0.0 + offset_y],
                [100.0 + offset_x, 0.0 + offset_y],
                [100.0 + offset_x, 100.0 + offset_y],
                [0.0 + offset_x, 100.0 + offset_y],
            ]
        })
        .collect()
}

/// 生成需要端点吸附的连续路径
/// 
/// 模拟真实 CAD 图纸中线段端点接近但不完全重合的情况
fn generate_snapping_path(num_segments: usize, length: f64, tolerance: f64) -> Vec<Vec<[f64; 2]>> {
    let mut segments = Vec::new();
    let mut current_x = 0.0;
    
    for _ in 0..num_segments {
        // 每个线段起点与前一个终点有微小偏移
        let start_x = current_x + tolerance * 0.3;
        let end_x = start_x + length;
        
        segments.push(vec![
            [start_x, 0.0],
            [end_x, 0.0],
        ]);
        
        current_x = end_x;
    }
    
    segments
}

/// 生成带孔洞的多边形数据
fn generate_polygon_with_holes(num_outer: usize, holes_per_polygon: usize) -> Vec<Vec<[f64; 2]>> {
    let mut polylines = Vec::new();
    
    for i in 0..num_outer {
        let offset = i as f64 * 500.0;
        
        // 外轮廓（200x200 正方形）
        let outer = vec![
            [0.0 + offset, 0.0],
            [200.0 + offset, 0.0],
            [200.0 + offset, 200.0],
            [0.0 + offset, 200.0],
        ];
        polylines.push(outer);
        
        // 内孔洞
        for j in 0..holes_per_polygon {
            let hole_offset_x = offset + 50.0 + (j as f64 * 60.0);
            let hole = vec![
                [hole_offset_x, 50.0],
                [hole_offset_x + 40.0, 50.0],
                [hole_offset_x + 40.0, 90.0],
                [hole_offset_x, 90.0],
            ];
            polylines.push(hole);
        }
    }
    
    polylines
}

/// 基准测试：小规模矩形网格（100 个矩形 = 400 条线段）
fn bench_topology_small(c: &mut Criterion) {
    let mut group = c.benchmark_group("topology_small");
    let polylines = generate_rectangles(100, 150.0);

    group.throughput(Throughput::Elements(400));
    group.bench_with_input(BenchmarkId::from_parameter("100 矩形"), &polylines, |b, polylines| {
        b.iter(|| {
            let topo = TopoService::with_default_config();
            let result = topo.build_topology(black_box(polylines));
            assert!(result.is_ok());
            result.unwrap().all_loops.len()
        })
    });
    group.finish();
}

/// 基准测试：中等规模矩形网格（1000 个矩形 = 4000 条线段）
fn bench_topology_medium(c: &mut Criterion) {
    let mut group = c.benchmark_group("topology_medium");
    let polylines = generate_rectangles(1000, 150.0);

    group.throughput(Throughput::Elements(4000));
    group.bench_with_input(BenchmarkId::from_parameter("1000 矩形"), &polylines, |b, polylines| {
        b.iter(|| {
            let topo = TopoService::with_default_config();
            let result = topo.build_topology(black_box(polylines));
            assert!(result.is_ok());
            result.unwrap().all_loops.len()
        })
    });
    group.finish();
}

/// 基准测试：大规模矩形网格（10000 个矩形 = 40000 条线段）
fn bench_topology_large(c: &mut Criterion) {
    let mut group = c.benchmark_group("topology_large");
    let polylines = generate_rectangles(10000, 150.0);

    group.throughput(Throughput::Elements(40000));
    group.bench_with_input(BenchmarkId::from_parameter("10000 矩形"), &polylines, |b, polylines| {
        b.iter(|| {
            let topo = TopoService::with_default_config();
            let result = topo.build_topology(black_box(polylines));
            assert!(result.is_ok());
            result.unwrap().all_loops.len()
        })
    });
    group.finish();
}

/// 基准测试：端点吸附场景
fn bench_topology_snapping(c: &mut Criterion) {
    let mut group = c.benchmark_group("topology_snapping");

    for &num_segments in &[100, 500, 1000, 5000, 10000] {
        let polylines = generate_snapping_path(num_segments, 10.0, 0.2);

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{} 线段", num_segments)),
            &polylines,
            |b, polylines| {
                b.iter(|| {
                    let topo = TopoService::with_default_config();
                    let result = topo.build_topology(black_box(polylines));
                    assert!(result.is_ok());
                    result.unwrap().all_loops.len()
                })
            }
        );
    }

    group.finish();
}

/// 基准测试：超大规模矩形网格（10000 个矩形 = 40000 条线段）
fn bench_topology_xlarge(c: &mut Criterion) {
    let mut group = c.benchmark_group("topology_xlarge");
    let polylines = generate_rectangles(10000, 150.0);

    group.throughput(Throughput::Elements(40000));
    group.bench_with_input(BenchmarkId::from_parameter("10000 矩形"), &polylines, |b, polylines| {
        b.iter(|| {
            let topo = TopoService::with_default_config();
            let result = topo.build_topology(black_box(polylines));
            result.is_ok()
        })
    });

    group.finish();
}

/// 基准测试：100K 线段大规模场景（🔴 Principal Engineer 要求补充）
///
/// 模拟大型建筑图纸（如整层平面图）
/// 注意：此测试耗时较长，默认标记为 ignore，需手动运行
#[ignore = "耗时较长，需手动运行：cargo bench --bench topology_bench -- --ignored"]
fn bench_topology_100k_segments(c: &mut Criterion) {
    let mut group = c.benchmark_group("topology_100k_segments");
    
    // 25000 个矩形 = 100000 条线段
    let polylines = generate_rectangles(25000, 150.0);

    group.throughput(Throughput::Elements(100000));
    group.bench_with_input(
        BenchmarkId::from_parameter("100K 线段 (25K 矩形)"),
        &polylines,
        |b, polylines| {
            b.iter(|| {
                let topo = TopoService::with_default_config();
                let result = topo.build_topology(black_box(polylines));
                assert!(result.is_ok());
                result.unwrap().all_loops.len()
            })
        }
    );

    group.finish();
}

/// 基准测试：500K 线段超大规模场景（🔴 Principal Engineer 要求补充）
///
/// 模拟园区级总平面图
/// 注意：此测试非常耗时，默认标记为 ignore，需手动运行
#[ignore = "非常耗时，需手动运行：cargo bench --bench topology_bench -- --ignored"]
fn bench_topology_500k_segments(c: &mut Criterion) {
    let mut group = c.benchmark_group("topology_500k_segments");
    
    // 125000 个矩形 = 500000 条线段
    let polylines = generate_rectangles(125000, 150.0);

    group.throughput(Throughput::Elements(500000));
    group.bench_with_input(
        BenchmarkId::from_parameter("500K 线段 (125K 矩形)"),
        &polylines,
        |b, polylines| {
            b.iter(|| {
                let topo = TopoService::with_default_config();
                let result = topo.build_topology(black_box(polylines));
                assert!(result.is_ok());
                result.unwrap().all_loops.len()
            })
        }
    );

    group.finish();
}

/// 基准测试：1M 线段极端场景（🔴 Principal Engineer 要求补充）
///
/// 极限压力测试，验证内存稳定性和性能边界
/// 注意：此测试极其耗时，默认标记为 ignore，需手动运行
#[ignore = "极其耗时，需手动运行：cargo bench --bench topology_bench -- --ignored"]
fn bench_topology_1m_segments(c: &mut Criterion) {
    let mut group = c.benchmark_group("topology_1m_segments");
    
    // 250000 个矩形 = 1000000 条线段
    let polylines = generate_rectangles(250000, 150.0);

    group.throughput(Throughput::Elements(1000000));
    group.bench_with_input(
        BenchmarkId::from_parameter("1M 线段 (250K 矩形)"),
        &polylines,
        |b, polylines| {
            b.iter(|| {
                let topo = TopoService::with_default_config();
                let result = topo.build_topology(black_box(polylines));
                assert!(result.is_ok());
                result.unwrap().all_loops.len()
            })
        }
    );

    group.finish();
}

/// 基准测试：带孔洞的多边形
fn bench_topology_with_holes(c: &mut Criterion) {
    let mut group = c.benchmark_group("topology_with_holes");

    for &(num_polygons, holes) in &[(10, 2), (50, 3), (100, 4)] {
        let total_polylines = num_polygons * (1 + holes);
        let polylines = generate_polygon_with_holes(num_polygons, holes);

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{} 多边形+{} 孔", num_polygons, holes)),
            &polylines,
            |b, polylines| {
                b.iter(|| {
                    let topo = TopoService::with_default_config();
                    let result = topo.build_topology(black_box(polylines));
                    assert!(result.is_ok());
                    result.unwrap().all_loops.len()
                })
            }
        );

        group.throughput(Throughput::Elements(total_polylines as u64));
    }

    group.finish();
}

/// 基准测试：容差敏感性分析
fn bench_topology_tolerance_sensitivity(c: &mut Criterion) {
    let mut group = c.benchmark_group("topology_tolerance_sensitivity");

    let polylines = generate_rectangles(500, 150.0);

    for &tolerance in &[0.1, 0.5, 1.0, 2.0] {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("容差={}", tolerance)),
            &tolerance,
            |b, &tolerance| {
                b.iter(|| {
                    let config = TopoConfig {
                        tolerance: ToleranceConfig {
                            snap_tolerance: tolerance,
                            min_line_length: 1.0,
                            max_angle_deviation: 5.0,
                            units: Some(LengthUnit::Mm),
                        },
                        layer_filter: None,
                        use_halfedge: false,
                    };
                    let topo = TopoService::new(config);
                    let result = topo.build_topology(black_box(&polylines));
                    assert!(result.is_ok());
                    result.unwrap().all_loops.len()
                })
            }
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_topology_small,
    bench_topology_medium,
    bench_topology_large,
    bench_topology_xlarge,
    bench_topology_100k_segments,
    bench_topology_500k_segments,
    bench_topology_1m_segments,
    bench_topology_snapping,
    bench_topology_with_holes,
    bench_topology_tolerance_sensitivity,
);

criterion_main!(benches);
