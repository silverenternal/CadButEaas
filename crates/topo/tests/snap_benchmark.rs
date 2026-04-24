//! 并行 snap 对比测试
use common_types::Point2;
use std::time::Instant;
use topo::parallel::snap_endpoints_parallel;

/// 生成测试点（网格状，无重复）
fn generate_points(n: usize) -> Vec<Point2> {
    let cols = ((n as f64).sqrt() as usize).max(1);
    (0..n)
        .map(|i| {
            let row = i / cols;
            let col = i % cols;
            [col as f64 * 10.0, row as f64 * 10.0]
        })
        .collect()
}

#[test]
fn benchmark_parallel_snap() {
    let sizes = vec![1_000, 10_000, 100_000];

    println!("\n=== 并行 Snap 性能测试 ===");
    println!("{:>10} | {:>12} | {:>12}", "Size", "Time", "Output");
    println!("{:-<40}", "");

    for size in &sizes {
        let points = generate_points(*size);

        let t0 = Instant::now();
        let (result, snap_index) = snap_endpoints_parallel(&points, 0.5);
        let elapsed = t0.elapsed();

        println!(
            "{:>10} | {:>10.2}ms | {:>12} | {:>12}",
            size,
            elapsed.as_secs_f64() * 1000.0,
            result.len(),
            snap_index.len()
        );
    }
}
