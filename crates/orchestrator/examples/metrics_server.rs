// orchestrator/examples/metrics_server.rs
// Prometheus 指标暴露示例 - 演示如何在应用中暴露指标

use std::net::SocketAddr;

use axum::{http::StatusCode, response::IntoResponse, routing::get, Router};
use prometheus::{Encoder, TextEncoder};
use tokio::net::TcpListener;
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;

/// 获取 Prometheus 指标
async fn metrics() -> impl IntoResponse {
    let encoder = TextEncoder::new();
    let metric_families = prometheus::gather();

    let mut buffer = Vec::new();
    match encoder.encode(&metric_families, &mut buffer) {
        Ok(_) => {
            // 检查指标内容
            let metrics_string = String::from_utf8_lossy(&buffer);

            // 添加自定义指标（示例）
            let response = format!(
                "{}\n# 自定义指标示例\n# cad_eaas_version_info 1.0.0\ncad_eaas_version_info{{version=\"1.0.0\"}} 1\n",
                metrics_string
            );

            (StatusCode::OK, response)
        }
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("无法编码指标：{}", e),
        ),
    }
}

/// 健康检查端点
async fn health() -> impl IntoResponse {
    (StatusCode::OK, "OK")
}

/// 就绪检查端点
async fn ready() -> impl IntoResponse {
    (StatusCode::OK, "Ready")
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 初始化日志
    FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .with_target(false)
        .init();

    info!("启动 CAD EaaS 指标服务器...");

    // 创建路由
    let app = Router::new()
        .route("/metrics", get(metrics))
        .route("/health", get(health))
        .route("/ready", get(ready));

    // 绑定地址
    let addr: SocketAddr = "127.0.0.1:8080".parse()?;
    let listener = TcpListener::bind(addr).await?;

    info!("指标服务器监听地址：http://{}", addr);
    info!("Prometheus 指标端点：http://{}/metrics", addr);
    info!("健康检查端点：http://{}/health", addr);
    info!("就绪检查端点：http://{}/ready", addr);
    tracing::info!("");
    tracing::info!("测试命令:");
    tracing::info!("  curl http://{}/metrics", addr);
    tracing::info!("  curl http://{}/health", addr);
    tracing::info!("");
    tracing::info!("按 Ctrl+C 停止服务器");

    // 启动服务器
    axum::serve(listener, app).await?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::Request;
    use tower::ServiceExt;

    #[tokio::test]
    async fn test_health_endpoint() {
        let app = Router::new()
            .route("/health", get(health))
            .route("/ready", get(ready))
            .route("/metrics", get(metrics));

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/health")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_ready_endpoint() {
        let app = Router::new()
            .route("/health", get(health))
            .route("/ready", get(ready))
            .route("/metrics", get(metrics));

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/ready")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_metrics_endpoint() {
        let app = Router::new()
            .route("/health", get(health))
            .route("/ready", get(ready))
            .route("/metrics", get(metrics));

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/metrics")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);

        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let body_str = String::from_utf8_lossy(&body);

        // 验证包含 Prometheus 指标格式
        assert!(body_str.contains("# HELP") || body_str.contains("cad_eaas"));
    }
}
