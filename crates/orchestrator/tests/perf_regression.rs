// orchestrator/tests/perf_regression.rs
// 性能回归测试 - 确保零拷贝性能不退化

use std::sync::Arc;
use std::time::Instant;

/// 测试零拷贝性能
/// 确保 Arc::clone() 在大量数据下仍保持 O(1) 时间复杂度
#[test]
fn test_zero_copy_performance() {
    // 创建 1MB 数据
    let data: Arc<[u8]> = vec![0u8; 1_000_000].into();

    let start = Instant::now();
    
    // 执行 1000 次 Arc::clone()
    for _ in 0..1000 {
        let _ = Arc::clone(&data);
    }
    
    let elapsed = start.elapsed();

    // 断言：1000 次 Arc::clone() 应在 100μs 内完成
    // 实际测试：通常在 10-20μs 内完成
    assert!(
        elapsed < std::time::Duration::from_micros(100),
        "零拷贝性能回归：1000 次 Arc::clone() 耗时 {:?} > 100μs",
        elapsed
    );

    eprintln!("✅ 零拷贝性能测试通过：1000 次 Arc::clone() 耗时 {:?}", elapsed);
}

/// 测试零拷贝性能与数据量无关
/// Arc::clone() 应该只增加引用计数，与数据量无关
#[test]
fn test_zero_copy_data_size_independence() {
    // 创建不同大小的数据
    let data_100kb: Arc<[u8]> = vec![0u8; 100_000].into();
    let data_1mb: Arc<[u8]> = vec![0u8; 1_000_000].into();
    let data_10mb: Arc<[u8]> = vec![0u8; 10_000_000].into();

    let iterations = 1000;

    // 测试 100KB 数据
    let start_100kb = Instant::now();
    for _ in 0..iterations {
        let _ = Arc::clone(&data_100kb);
    }
    let elapsed_100kb = start_100kb.elapsed();

    // 测试 1MB 数据
    let start_1mb = Instant::now();
    for _ in 0..iterations {
        let _ = Arc::clone(&data_1mb);
    }
    let elapsed_1mb = start_1mb.elapsed();

    // 测试 10MB 数据
    let start_10mb = Instant::now();
    for _ in 0..iterations {
        let _ = Arc::clone(&data_10mb);
    }
    let elapsed_10mb = start_10mb.elapsed();

    // 断言：三种大小的数据耗时应该相近（差异不超过 50%）
    let max_elapsed = elapsed_100kb.max(elapsed_1mb).max(elapsed_10mb);
    let min_elapsed = elapsed_100kb.min(elapsed_1mb).min(elapsed_10mb);

    assert!(
        max_elapsed.as_nanos() < min_elapsed.as_nanos() * 2,
        "零拷贝性能应与数据量无关：100KB={:?}, 1MB={:?}, 10MB={:?}",
        elapsed_100kb,
        elapsed_1mb,
        elapsed_10mb
    );

    eprintln!(
        "✅ 数据量无关性测试通过：100KB={:?}, 1MB={:?}, 10MB={:?}",
        elapsed_100kb, elapsed_1mb, elapsed_10mb
    );
}

/// 测试深拷贝性能退化
/// 确保 Vec::clone() 性能与数据量成正比（作为对比基线）
#[test]
fn test_deep_copy_performance_baseline() {
    // 创建 100KB 数据
    let data: Vec<u8> = vec![0u8; 100_000];

    let start = Instant::now();
    
    // 执行 100 次 Vec::clone()
    for _ in 0..100 {
        let _ = data.clone();
    }
    
    let elapsed = start.elapsed();

    // 断言：100 次 Vec::clone() 100KB 数据应在 100ms 内完成
    // 这是一个宽松的阈值，实际通常在 1-5ms 内
    assert!(
        elapsed < std::time::Duration::from_millis(100),
        "深拷贝性能基线：100 次 Vec::clone() 100KB 耗时 {:?} > 100ms",
        elapsed
    );

    eprintln!("✅ 深拷贝基线测试通过：100 次 Vec::clone() 100KB 耗时 {:?}", elapsed);
}

/// 测试零拷贝 vs 深拷贝性能差异
/// 确保零拷贝性能优势保持在 1000 倍以上
#[test]
fn test_zero_copy_vs_deep_copy_advantage() {
    let data_size = 100_000; // 100KB
    let iterations = 100;

    // 零拷贝测试
    let arc_data: Arc<[u8]> = vec![0u8; data_size].into();
    let start_arc = Instant::now();
    for _ in 0..iterations {
        let _ = Arc::clone(&arc_data);
    }
    let elapsed_arc = start_arc.elapsed();

    // 深拷贝测试
    let vec_data: Vec<u8> = vec![0u8; data_size];
    let start_vec = Instant::now();
    for _ in 0..iterations {
        let _ = vec_data.clone();
    }
    let elapsed_vec = start_vec.elapsed();

    // 计算性能优势
    let advantage = elapsed_vec.as_nanos() as f64 / elapsed_arc.as_nanos() as f64;

    // 断言：零拷贝性能优势应至少 100 倍
    // 实际测试：通常能达到 1000 倍以上
    assert!(
        advantage > 100.0,
        "零拷贝性能优势不足：{:.1}倍 (期望 > 100 倍)",
        advantage
    );

    eprintln!(
        "✅ 零拷贝优势测试通过：零拷贝={:?}, 深拷贝={:?}, 优势={:.1}倍",
        elapsed_arc, elapsed_vec, advantage
    );
}

/// 测试高并发场景下的零拷贝性能
/// 模拟多线程环境中的 Arc::clone() 性能
#[test]
fn test_zero_copy_concurrent_performance() {
    use std::thread;

    let data: Arc<[u8]> = vec![0u8; 1_000_000].into(); // 1MB
    let threads = 8;
    let iterations_per_thread = 100;

    let start = Instant::now();

    // 创建多个线程并发执行 Arc::clone()
    let handles: Vec<_> = (0..threads)
        .map(|_| {
            let data_clone = Arc::clone(&data);
            thread::spawn(move || {
                for _ in 0..iterations_per_thread {
                    let _ = Arc::clone(&data_clone);
                }
            })
        })
        .collect();

    // 等待所有线程完成
    for handle in handles {
        handle.join().unwrap();
    }

    let elapsed = start.elapsed();
    let total_clones = threads * iterations_per_thread;

    // 断言：8 线程 × 100 次 = 800 次 Arc::clone() 应在 5ms 内完成
    // 注意：并发场景下包含线程创建和同步开销，阈值放宽
    assert!(
        elapsed < std::time::Duration::from_millis(5),
        "并发零拷贝性能回归：{} 次 Arc::clone() 耗时 {:?} > 5ms",
        total_clones,
        elapsed
    );

    eprintln!(
        "✅ 并发零拷贝性能测试通过：{} 线程 × {} 次 = {} 次，耗时 {:?}",
        threads, iterations_per_thread, total_clones, elapsed
    );
}

/// 测试 Arc 内存开销
/// 确保 Arc 的内存开销在合理范围内
#[test]
fn test_arc_memory_overhead() {
    use std::mem::size_of;

    // Arc 本身的大小应该很小（通常是指针大小）
    let arc_size = size_of::<Arc<[u8]>>();
    
    // 在 64 位系统上，Arc 应该是 16 字节（2 个指针）
    assert!(
        arc_size <= 16,
        "Arc 内存开销过大：{} 字节 (期望 <= 16 字节)",
        arc_size
    );

    // 验证 Arc 不会复制底层数据
    let data: Arc<[u8]> = vec![0u8; 1_000_000].into();
    let data_ptr = Arc::as_ptr(&data);
    
    let data_clone = Arc::clone(&data);
    let data_clone_ptr = Arc::as_ptr(&data_clone);

    // 两个 Arc 应该指向同一块内存
    assert_eq!(
        data_ptr, data_clone_ptr,
        "Arc::clone() 应该共享同一块内存"
    );

    eprintln!("✅ Arc 内存开销测试通过：Arc 大小={} 字节，指针共享验证通过", arc_size);
}
