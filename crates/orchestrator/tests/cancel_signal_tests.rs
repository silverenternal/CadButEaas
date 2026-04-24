/// 取消信号传播延迟测试
///
/// 验证 broadcast 通道的快速失败机制：
/// - 100 个接收者应在 10ms 内收到取消信号
/// - 验证"先订阅后执行"模式的正确性
use std::time::{Duration, Instant};
use tokio::sync::broadcast;

#[tokio::test]
async fn test_cancel_signal_propagation_100_rx() {
    // 创建容量充足的 broadcast 通道
    let (cancel_tx, _) = broadcast::channel::<()>(100);

    // 创建 100 个接收者（模拟 100 个并行任务）
    let mut rxs = Vec::new();
    for _ in 0..100 {
        rxs.push(cancel_tx.subscribe());
    }

    let start = Instant::now();

    // 发送取消信号
    cancel_tx.send(()).expect("发送取消信号失败");

    // 等待所有接收者收到信号
    for mut rx in rxs {
        tokio::time::timeout(Duration::from_millis(10), rx.recv())
            .await
            .expect("接收超时")
            .expect("通道关闭");
    }

    let elapsed = start.elapsed();
    println!("取消信号传播延迟（100 个接收者）: {:?}", elapsed);

    // 断言：100 个接收者应在 10ms 内收到信号
    assert!(
        elapsed < Duration::from_millis(10),
        "取消信号传播延迟过高：{:?} > 10ms",
        elapsed
    );
}

#[tokio::test]
async fn test_cancel_signal_propagation_1000_rx() {
    // 创建容量充足的 broadcast 通道
    let (cancel_tx, _) = broadcast::channel::<()>(1000);

    // 创建 1000 个接收者（模拟大规模并行任务）
    let mut rxs = Vec::new();
    for _ in 0..1000 {
        rxs.push(cancel_tx.subscribe());
    }

    let start = Instant::now();

    // 发送取消信号
    cancel_tx.send(()).expect("发送取消信号失败");

    // 等待所有接收者收到信号
    for mut rx in rxs {
        tokio::time::timeout(Duration::from_millis(50), rx.recv())
            .await
            .expect("接收超时")
            .expect("通道关闭");
    }

    let elapsed = start.elapsed();
    println!("取消信号传播延迟（1000 个接收者）: {:?}", elapsed);

    // 断言：1000 个接收者应在 50ms 内收到信号
    assert!(
        elapsed < Duration::from_millis(50),
        "取消信号传播延迟过高：{:?} > 50ms",
        elapsed
    );
}

/// 测试"先订阅后执行"模式
#[tokio::test]
async fn test_subscribe_before_execute() {
    let (cancel_tx, _) = broadcast::channel::<()>(10);

    // 先创建所有接收者（模拟任务启动前先订阅）
    let mut rxs = Vec::new();
    for _ in 0..10 {
        rxs.push(cancel_tx.subscribe());
    }

    // 再发送取消信号（模拟某个任务失败后取消其他任务）
    cancel_tx.send(()).expect("发送取消信号失败");

    // 验证所有接收者都能收到信号
    for mut rx in rxs {
        tokio::time::timeout(Duration::from_millis(100), rx.recv())
            .await
            .expect("接收超时")
            .expect("通道关闭");
        // recv() 返回 Result<T, RecvError>，成功时直接是 T（这里是 ()）
        (); // 收到 () 单元类型
    }
}

/// 测试 SendError 处理（没有接收者时发送）
#[tokio::test]
async fn test_send_error_handling() {
    let (cancel_tx, _) = broadcast::channel::<()>(10);

    // 不创建接收者，直接发送
    match cancel_tx.send(()) {
        Ok(_) => panic!("应该返回 SendError"),
        Err(broadcast::error::SendError(_)) => {
            // 预期行为：没有接收者时返回 SendError
            println!("正确处理 SendError：没有接收者");
        }
    }
}

/// 测试容量不足时的行为
#[tokio::test]
async fn test_channel_capacity() {
    // 创建容量为 1 的通道
    let (cancel_tx, _) = broadcast::channel::<()>(1);

    // 创建 10 个接收者（超过容量）
    let mut rxs = Vec::new();
    for _ in 0..10 {
        rxs.push(cancel_tx.subscribe());
    }

    // 发送信号应该成功（broadcast 通道容量是"瞬时"的，不影响发送）
    let result = cancel_tx.send(());
    assert!(result.is_ok(), "发送应该成功");

    // 验证所有接收者都能收到信号
    for mut rx in rxs {
        tokio::time::timeout(Duration::from_millis(100), rx.recv())
            .await
            .expect("接收超时")
            .expect("通道关闭");
        // recv() 返回 Result<T, RecvError>，成功时直接是 T（这里是 ()）
        (); // 收到 () 单元类型
    }
}
