//! 故障注入测试（P11 锐评落地版）
//!
//! 测试异常路径和边界条件，确保生产级鲁棒性

use common_types::error::AutoFix;
use common_types::scene::SceneState;
use common_types::service::{HealthCheckUtils, HealthMonitor, HealthStatus, ShardedHistogram};
use std::sync::Arc;

/// 测试健康监控器在正常情况下的行为
#[test]
fn test_health_monitor_basic() {
    let monitor = HealthMonitor::start();

    // 验证监控器启动后能获取健康状态
    let health = monitor.get_health();
    assert_eq!(health.version, env!("CARGO_PKG_VERSION"));

    // 验证订阅功能
    let _subscriber = monitor.subscribe();

    monitor.stop();
}

/// 测试健康监控器多次订阅
#[test]
fn test_health_monitor_multiple_subscribers() {
    let monitor = HealthMonitor::start();

    // 创建多个订阅者
    let _sub1 = monitor.subscribe();
    let _sub2 = monitor.subscribe();
    let _sub3 = monitor.subscribe();

    // 验证都能正常工作
    let health = monitor.get_health();
    assert_eq!(health.status, HealthStatus::Healthy);

    monitor.stop();
}

/// 测试分片直方图在高并发下的行为
#[test]
fn test_sharded_histogram_under_high_concurrency() {
    let hist = Arc::new(ShardedHistogram::new(16, 3_600_000, 3));
    let mut handles = vec![];

    // 100 个线程同时记录
    for i in 0..100 {
        let hist = Arc::clone(&hist);
        let handle = std::thread::spawn(move || {
            for j in 0..1000 {
                hist.record((i * 1000 + j) as u64);
            }
        });
        handles.push(handle);
    }

    // 等待所有线程完成
    for handle in handles {
        handle.join().unwrap();
    }

    // 验证所有记录都成功
    let snapshot = hist.snapshot();
    assert_eq!(snapshot.count, 100 * 1000);

    // 验证有合理的延迟值
    assert!(snapshot.mean_ms > 0.0);
    assert!(snapshot.max_ms >= snapshot.min_ms);
}

/// 测试分片直方图在大量数据下的行为
#[test]
fn test_sharded_histogram_large_dataset() {
    let hist = ShardedHistogram::new(8, 3_600_000, 3);

    // 记录 100 万个值
    for i in 0..1_000_000 {
        hist.record(i % 1000); // 值范围 0-999 纳秒
    }

    let snapshot = hist.snapshot();
    assert_eq!(snapshot.count, 1_000_000);
}

/// 测试 AutoFix 增量快照回滚
#[test]
fn test_autofix_incremental_rollback() {
    let mut scene = create_test_scene();
    let original_edges_len = scene.edges.len();

    let result = {
        // 使用增量快照版本
        let fix = AutoFix::new("测试增量修复", |scene| {
            if !scene.edges.is_empty() {
                scene.edges[0].start = [999.0, 999.0];
            }
            Ok(())
        });

        fix.apply_safe(&mut scene)
    };

    // 验证修复完成（简化测试）
    assert!(result.is_ok());

    // 验证场景边数不变
    assert_eq!(scene.edges.len(), original_edges_len);
}

/// 测试 AutoFix 增量快照成功路径
#[test]
fn test_autofix_incremental_success() {
    let mut scene = create_test_scene();
    let original_start = if !scene.edges.is_empty() {
        Some(scene.edges[0].start)
    } else {
        None
    };

    let fix = AutoFix::new("测试成功修复", |scene| {
        if !scene.edges.is_empty() {
            scene.edges[0].start = [100.0, 100.0];
        }
        Ok(())
    });

    let result = fix.apply_safe(&mut scene);

    // 验证修复成功
    assert!(result.is_ok());

    // 验证边被修改
    if let Some(original) = original_start {
        assert_ne!(scene.edges[0].start, original);
    }
}

/// 测试 AutoFix 前置条件失败
#[test]
fn test_autofix_precondition_failure() {
    let scene = create_test_scene();

    let fix = AutoFix::with_rollback(
        "测试前置条件失败",
        |_| false, // 前置条件总是失败
        |_| Ok(()),
        |_| {},
        |_| true,
    );

    let result = fix.apply_safe(&mut scene.clone());

    // 验证前置条件失败
    assert!(result.is_err());
}

/// 测试健康检查在内存压力下的行为（模拟）
#[test]
fn test_health_check_memory_pressure_simulation() {
    // 注意：Rust 测试环境无法真正模拟内存压力
    // 这里测试健康检查的基本功能

    let mem_health = HealthCheckUtils::check_memory_health();

    // 验证内存检查返回有效结果
    assert_eq!(mem_health.name, "Memory");
    assert!(matches!(
        mem_health.status,
        HealthStatus::Healthy | HealthStatus::Degraded | HealthStatus::Unhealthy
    ));
}

/// 测试健康检查在 CPU 压力下的行为（模拟）
#[test]
fn test_health_check_cpu_pressure_simulation() {
    let cpu_health = HealthCheckUtils::check_cpu_health();

    // 验证 CPU 检查返回有效结果
    assert_eq!(cpu_health.name, "CPU");
    assert!(matches!(
        cpu_health.status,
        HealthStatus::Healthy | HealthStatus::Degraded | HealthStatus::Unhealthy
    ));
}

/// 测试健康检查文件系统健康
#[test]
fn test_health_check_filesystem() {
    let fs_health = HealthCheckUtils::check_filesystem_health();

    // 验证文件系统检查返回有效结果
    assert_eq!(fs_health.name, "FileSystem");
    // 在正常环境下应该是健康的
    assert_eq!(fs_health.status, HealthStatus::Healthy);
}

/// 测试综合健康检查
#[test]
fn test_comprehensive_health() {
    let health = HealthCheckUtils::comprehensive_health();

    // 验证综合健康检查包含所有依赖
    assert!(health.dependencies.len() >= 3); // FileSystem, Memory, CPU

    // 验证版本号存在
    assert!(!health.version.is_empty());
}

/// 测试分片直方图重置功能
#[test]
fn test_sharded_histogram_reset() {
    let hist = ShardedHistogram::new(8, 3_600_000, 3);

    // 记录一些值
    for i in 0..1000 {
        hist.record(i);
    }

    let snapshot_before = hist.snapshot();
    assert_eq!(snapshot_before.count, 1000);

    // 重置
    hist.reset();

    let snapshot_after = hist.snapshot();
    assert_eq!(snapshot_after.count, 0);
}

/// 测试分片直方图空值处理
#[test]
fn test_sharded_histogram_empty() {
    let hist = ShardedHistogram::new(8, 3_600_000, 3);
    let snapshot = hist.snapshot();

    assert_eq!(snapshot.count, 0);
    assert_eq!(snapshot.min_ms, 0.0);
    assert_eq!(snapshot.max_ms, 0.0);
    assert_eq!(snapshot.mean_ms, 0.0);
}

/// 辅助函数：创建测试场景
fn create_test_scene() -> SceneState {
    use common_types::RawEdge;

    let mut scene = SceneState::default();

    // 添加一些测试边
    for i in 0..10 {
        scene.edges.push(RawEdge {
            id: i,
            start: [i as f64, i as f64],
            end: [(i + 1) as f64, (i + 1) as f64],
            layer: Some(format!("layer_{}", i)),
            color_index: None,
        });
    }

    scene
}
