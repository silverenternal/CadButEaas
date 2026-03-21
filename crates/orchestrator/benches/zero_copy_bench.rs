/// 零拷贝性能基准测试
///
/// 对比 Arc<[T]> 和 Vec<T> 的克隆性能：
/// - Arc::clone(): O(1)，只增加引用计数（~1ns）
/// - Vec::clone(): O(n)，深拷贝所有数据（~1ms for 1MB）
///
/// 预期结果：Arc 零拷贝性能提升 100 万倍
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use std::sync::Arc;

/// 测试 Arc<[u8]> 的克隆性能（零拷贝）
fn bench_arc_clone(c: &mut Criterion) {
    // 1MB 数据
    let data: Arc<[u8]> = vec![0u8; 1_000_000].into();

    c.bench_function("arc_clone_1mb", |b| {
        b.iter(|| {
            let _clone = Arc::clone(&data);
            black_box(_clone);
        })
    });
}

/// 测试 Arc<[u8]> 的克隆性能（10MB）
fn bench_arc_clone_10mb(c: &mut Criterion) {
    let data: Arc<[u8]> = vec![0u8; 10_000_000].into();

    c.bench_function("arc_clone_10mb", |b| {
        b.iter(|| {
            let _clone = Arc::clone(&data);
            black_box(_clone);
        })
    });
}

/// 测试 Vec<u8> 的克隆性能（深拷贝，1MB）
fn bench_vec_clone(c: &mut Criterion) {
    let data: Vec<u8> = vec![0u8; 1_000_000];

    c.bench_function("vec_clone_1mb", |b| {
        b.iter(|| {
            let _clone = data.clone();
            black_box(_clone);
        })
    });
}

/// 测试 Vec<u8> 的克隆性能（深拷贝，100KB）
fn bench_vec_clone_100kb(c: &mut Criterion) {
    let data: Vec<u8> = vec![0u8; 100_000];

    c.bench_function("vec_clone_100kb", |b| {
        b.iter(|| {
            let _clone = data.clone();
            black_box(_clone);
        })
    });
}

/// 测试 Arc<[RawEntity]> 的克隆性能（模拟真实场景）
fn bench_arc_clone_entities(c: &mut Criterion) {
    // 模拟 1000 个实体
    let entities: Arc<[u8]> = vec![0u8; 1000 * 64].into(); // 假设每个实体 64 字节

    c.bench_function("arc_clone_1000_entities", |b| {
        b.iter(|| {
            let _clone = Arc::clone(&entities);
            black_box(_clone);
        })
    });
}

criterion_group!(
    benches,
    bench_arc_clone,
    bench_arc_clone_10mb,
    bench_vec_clone,
    bench_vec_clone_100kb,
    bench_arc_clone_entities,
);

criterion_main!(benches);
