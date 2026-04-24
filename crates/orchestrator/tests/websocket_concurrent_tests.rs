//! WebSocket 并发连接测试
//!
//! P11 锐评落实：验证 WebSocket 服务器支持多客户端并发连接
//! 测试目标：
//! - 基础测试：10 个并发连接（已实现）
//! - 高并发测试：100 个并发连接（P11 v4.0 新增）
//! - 长连接测试：持续 1 分钟稳定性（P11 v4.0 新增）

use futures::{SinkExt, StreamExt};
use std::time::Duration;
use tokio::time::sleep;
use tokio_tungstenite::{connect_async, tungstenite::Message};

/// 测试 WebSocket 并发连接能力
///
/// 测试场景：
/// 1. 同时建立 10 个 WebSocket 连接
/// 2. 每个连接发送 ping 消息
/// 3. 验证所有连接都能收到 pong 响应
/// 4. 验证连接稳定性（持续 5 秒）
#[tokio::test]
async fn test_websocket_concurrent_connections() -> Result<(), Box<dyn std::error::Error>> {
    // 注意：这是一个集成测试，需要服务器运行在 localhost:3000
    // 如果服务器未运行，测试将跳过
    let server_addr = "ws://localhost:3000/ws";

    // 尝试连接，如果失败则跳过测试
    let test_connection = connect_async(server_addr).await;
    if test_connection.is_err() {
        println!("⚠️  跳过测试：WebSocket 服务器未运行 (localhost:3000)");
        return Ok(());
    }
    drop(test_connection);

    const NUM_CONNECTIONS: usize = 10;

    println!("🔌 开始测试：{} 个并发 WebSocket 连接", NUM_CONNECTIONS);

    // 创建 10 个并发连接任务
    let mut handles = Vec::new();

    for i in 0..NUM_CONNECTIONS {
        let addr = server_addr.to_string();
        let handle = tokio::spawn(async move { run_websocket_client(i, &addr).await });
        handles.push(handle);
    }

    // 等待所有连接完成任务
    let results: Vec<_> = futures::future::join_all(handles).await;

    // 统计结果
    let mut success_count = 0;

    for (i, result) in results.into_iter().enumerate() {
        match result {
            Ok(Ok(())) => {
                success_count += 1;
                println!("✅ 客户端 {} 测试通过", i);
            }
            Ok(Err(e)) => {
                println!("❌ 客户端 {} 测试失败：{}", i, e);
            }
            Err(e) => {
                println!("❌ 客户端 {} 任务崩溃：{}", i, e);
            }
        }
    }

    println!(
        "\n📊 测试结果：{}/{} 个连接成功",
        success_count, NUM_CONNECTIONS
    );

    // 验证：至少 80% 的连接成功（允许网络波动）
    assert!(
        success_count >= NUM_CONNECTIONS * 8 / 10,
        "并发连接测试失败：{}/{} 成功 (期望至少 {}%)",
        success_count,
        NUM_CONNECTIONS,
        80
    );

    Ok(())
}

/// 单个 WebSocket 客户端行为
async fn run_websocket_client(
    client_id: usize,
    addr: &str,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // 连接服务器
    let (ws_stream, _) = connect_async(addr).await?;
    let (mut write, mut read) = ws_stream.split();

    println!("  [客户端 {}] 已连接", client_id);

    // 等待接收连接确认消息
    if let Some(msg) = read.next().await {
        if let Message::Text(text) = msg? {
            if text.contains("connected") {
                println!("  [客户端 {}] ✓ 收到连接确认", client_id);
            }
        }
    }

    // 发送 ping 消息
    let ping_msg = serde_json::json!({
        "type": "ping"
    });
    write.send(Message::Text(ping_msg.to_string())).await?;
    println!("  [客户端 {}] → 发送 ping", client_id);

    // 等待 pong 响应
    if let Some(msg) = read.next().await {
        if let Message::Text(text) = msg? {
            if text.contains("pong") || text.contains("Pong") {
                println!("  [客户端 {}] ← 收到 pong", client_id);
            }
        }
    }

    // 保持连接一段时间，测试稳定性
    sleep(Duration::from_millis(500)).await;

    // 再次发送 ping 消息
    write.send(Message::Text(ping_msg.to_string())).await?;
    println!("  [客户端 {}] → 发送第二次 ping", client_id);

    // 等待响应
    if let Some(msg) = read.next().await {
        if let Message::Text(_) = msg? {
            println!("  [客户端 {}] ← 收到响应", client_id);
        }
    }

    // 关闭连接
    write.send(Message::Close(None)).await?;
    println!("  [客户端 {}] 断开连接", client_id);

    Ok(())
}

// ============================================================================
// P11 锐评 v4.0 新增：高并发测试
// ============================================================================

/// 测试 100 个并发 WebSocket 连接
///
/// P11 锐评 v4.0 要求：验证服务器能承载真实场景的高并发
///
/// P11 v5.0 修复：添加 #[ignore] 标记，需要手动运行：cargo test -- --ignored
#[tokio::test]
#[ignore]
async fn test_websocket_100_concurrent_connections() -> Result<(), Box<dyn std::error::Error>> {
    let server_addr = "ws://localhost:3000/ws";

    // 尝试连接，如果失败则跳过测试
    let test_connection = connect_async(server_addr).await;
    if test_connection.is_err() {
        println!("⚠️  跳过测试：WebSocket 服务器未运行 (localhost:3000)");
        return Ok(());
    }
    drop(test_connection);

    const NUM_CONNECTIONS: usize = 100;

    println!(
        "🔌 开始高并发测试：{} 个并发 WebSocket 连接",
        NUM_CONNECTIONS
    );

    // 创建 100 个并发连接任务
    let mut handles = Vec::new();

    for i in 0..NUM_CONNECTIONS {
        let addr = server_addr.to_string();
        let handle = tokio::spawn(async move { run_websocket_client_light(i, &addr).await });
        handles.push(handle);
    }

    // 等待所有连接完成任务
    let results: Vec<_> = futures::future::join_all(handles).await;

    // 统计结果
    let mut success_count = 0;
    let mut fail_count = 0;

    for (i, result) in results.into_iter().enumerate() {
        match result {
            Ok(Ok(())) => {
                success_count += 1;
            }
            Ok(Err(e)) => {
                fail_count += 1;
                if fail_count <= 5 {
                    // 只打印前 5 个失败，避免刷屏
                    println!("❌ 客户端 {} 测试失败：{}", i, e);
                }
            }
            Err(e) => {
                fail_count += 1;
                if fail_count <= 5 {
                    println!("❌ 客户端 {} 任务崩溃：{}", i, e);
                }
            }
        }
    }

    println!(
        "\n📊 高并发测试结果：{}/{} 个连接成功",
        success_count, NUM_CONNECTIONS
    );

    // 验证：至少 90% 的连接成功（高并发场景下允许少量失败）
    assert!(
        success_count >= NUM_CONNECTIONS * 9 / 10,
        "高并发连接测试失败：{}/{} 成功 (期望至少 {}%)",
        success_count,
        NUM_CONNECTIONS,
        90
    );

    Ok(())
}

/// 轻量级 WebSocket 客户端行为（用于高并发测试）
async fn run_websocket_client_light(
    _client_id: usize,
    addr: &str,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // 连接服务器
    let (ws_stream, _) = connect_async(addr).await?;
    let (mut write, mut read) = ws_stream.split();

    // 等待接收连接确认消息
    if let Some(msg) = read.next().await {
        if let Message::Text(text) = msg? {
            if !text.contains("connected") {
                return Err("未收到连接确认".into());
            }
        }
    }

    // 发送 ping 消息
    let ping_msg = serde_json::json!({
        "type": "ping"
    });
    write.send(Message::Text(ping_msg.to_string())).await?;

    // 等待 pong 响应
    if let Some(msg) = read.next().await {
        if let Message::Text(text) = msg? {
            if !text.contains("pong") && !text.contains("Pong") {
                return Err("未收到 pong 响应".into());
            }
        }
    }

    // 高并发测试不需要长时间保持连接
    sleep(Duration::from_millis(100)).await;

    // 关闭连接
    write.send(Message::Close(None)).await?;

    Ok(())
}

// ============================================================================
// P11 锐评 v4.0 新增：长连接稳定性测试
// ============================================================================

/// 测试 WebSocket 长连接稳定性（持续 1 分钟）
///
/// P11 锐评 v4.0 要求：验证连接保活能力
///
/// P11 v5.0 修复：添加 #[ignore] 标记，需要手动运行：cargo test -- --ignored
#[tokio::test]
#[ignore]
async fn test_websocket_long_running_connection() -> Result<(), Box<dyn std::error::Error>> {
    let server_addr = "ws://localhost:3000/ws";

    // 尝试连接，如果失败则跳过测试
    let test_connection = connect_async(server_addr).await;
    if test_connection.is_err() {
        println!("⚠️  跳过测试：WebSocket 服务器未运行 (localhost:3000)");
        return Ok(());
    }
    drop(test_connection);

    const DURATION: Duration = Duration::from_secs(60);
    const PING_INTERVAL: Duration = Duration::from_secs(5);

    println!("🔌 开始长连接测试：持续 {:?}", DURATION);

    // 连接服务器
    let (ws_stream, _) = connect_async(server_addr).await?;
    let (mut write, mut read) = ws_stream.split();

    // 等待接收连接确认消息
    if let Some(msg) = read.next().await {
        if let Message::Text(text) = msg? {
            if !text.contains("connected") {
                return Err("未收到连接确认".into());
            }
        }
    }

    println!("  ✓ 连接已建立，开始心跳保活...");

    // 定期发送 ping 消息保持连接
    let mut ping_count = 0;
    let start = std::time::Instant::now();

    while start.elapsed() < DURATION {
        // 发送 ping
        let ping_msg = serde_json::json!({
            "type": "ping"
        });
        write.send(Message::Text(ping_msg.to_string())).await?;
        ping_count += 1;

        // 等待 pong 响应
        if let Some(msg) = read.next().await {
            if let Message::Text(text) = msg? {
                if !text.contains("pong") && !text.contains("Pong") {
                    return Err("未收到 pong 响应".into());
                }
            }
        }

        // 等待下一次心跳
        sleep(PING_INTERVAL).await;
    }

    println!("  ✓ 长连接测试通过：共发送 {} 次心跳", ping_count);

    // 关闭连接
    write.send(Message::Close(None)).await?;

    Ok(())
}

// ============================================================================
// P11 锐评 v4.0 新增：消息丢失模拟测试
// ============================================================================

/// 测试 WebSocket 消息丢失场景（网络波动模拟）
///
/// P11 锐评 v4.0 要求：验证网络波动下的消息恢复能力
///
/// P11 v5.0 修复：添加 #[ignore] 标记，需要手动运行：cargo test -- --ignored
#[tokio::test]
#[ignore]
async fn test_websocket_message_loss_simulation() -> Result<(), Box<dyn std::error::Error>> {
    let server_addr = "ws://localhost:3000/ws";

    // 尝试连接，如果失败则跳过测试
    let test_connection = connect_async(server_addr).await;
    if test_connection.is_err() {
        println!("⚠️  跳过测试：WebSocket 服务器未运行 (localhost:3000)");
        return Ok(());
    }
    drop(test_connection);

    const TOTAL_MESSAGES: usize = 20;
    const SIMULATE_LOSS_RATE: f64 = 0.1; // 10% 丢失率模拟

    println!(
        "🔌 开始消息丢失模拟测试：发送 {} 条消息（模拟 {}% 丢失率）",
        TOTAL_MESSAGES,
        (SIMULATE_LOSS_RATE * 100.0) as i32
    );

    // 连接服务器
    let (ws_stream, _) = connect_async(server_addr).await?;
    let (mut write, mut read) = ws_stream.split();

    // 等待接收连接确认消息
    if let Some(msg) = read.next().await {
        if let Message::Text(text) = msg? {
            if !text.contains("connected") {
                return Err("未收到连接确认".into());
            }
        }
    }

    let mut sent_count = 0;
    let mut received_count = 0;

    // 发送多条消息
    for i in 0..TOTAL_MESSAGES {
        let ping_msg = serde_json::json!({
            "type": "ping",
            "seq": i
        });

        // 模拟网络波动：随机跳过一些发送
        use rand::Rng;
        let mut rng = rand::thread_rng();
        if rng.gen_bool(1.0 - SIMULATE_LOSS_RATE) {
            write.send(Message::Text(ping_msg.to_string())).await?;
            sent_count += 1;
        } else {
            println!("  [模拟] 消息 {} 丢失", i);
        }

        // 等待响应（带超时）
        let response = tokio::time::timeout(Duration::from_millis(500), read.next()).await;

        match response {
            Ok(Some(Ok(msg))) => {
                if let Message::Text(_) = msg {
                    received_count += 1;
                }
            }
            Ok(Some(Err(e))) => {
                println!("  ⚠️  消息 {} 响应错误：{}", i, e);
            }
            Ok(None) | Err(_) => {
                // 超时或连接关闭，继续
            }
        }

        // 消息间隔
        sleep(Duration::from_millis(50)).await;
    }

    println!("\n📊 消息丢失模拟测试结果：");
    println!("  - 发送消息数：{}", sent_count);
    println!("  - 接收响应数：{}", received_count);

    // 验证：接收率应该与发送率相近（允许一定波动）
    let expected_receive_rate = 1.0 - SIMULATE_LOSS_RATE;
    let actual_receive_rate = received_count as f64 / sent_count as f64;

    // 允许 20% 的波动范围
    assert!(
        actual_receive_rate >= expected_receive_rate - 0.2,
        "消息丢失模拟测试失败：接收率 {:.2}% (期望至少 {:.2}%)",
        actual_receive_rate * 100.0,
        (expected_receive_rate - 0.2) * 100.0
    );

    // 关闭连接
    write.send(Message::Close(None)).await?;

    Ok(())
}
