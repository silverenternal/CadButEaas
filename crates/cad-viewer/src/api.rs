//! HTTP API 客户端 - 与后端 orchestrator 交互（P11 落实版：WebSocket 实时交互）

use interact::Edge;
use log::info;
use reqwest::multipart;
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::Arc;
use tokio::sync::Mutex;
use futures::{SinkExt, StreamExt};
use tokio_tungstenite::{connect_async, tungstenite::Message};

// WebSocket 类型别名
type WsStream = tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>;

/// API 客户端
#[derive(Clone)]
pub struct ApiClient {
    base_url: String,
    client: reqwest::Client,
    /// WebSocket 连接（P11 新增：实时交互）
    ws_connection: Arc<Mutex<Option<WsStream>>>,
    /// WebSocket 是否已连接
    ws_connected: Arc<Mutex<bool>>,
}

/// 自动追踪请求
#[derive(Serialize)]
pub struct AutoTraceRequest {
    pub edge_id: usize,
}

/// 自动追踪响应
#[derive(Deserialize, Clone)]
#[allow(dead_code)] // 预留用于未来功能
pub struct AutoTraceResponse {
    pub success: bool,
    pub loop_points: Option<Vec<[f64; 2]>>,
    pub message: String,
}

/// 圈选请求
#[derive(Serialize)]
#[allow(dead_code)] // 预留用于未来功能
pub struct LassoRequest {
    pub polygon: Vec<[f64; 2]>,
}

/// 圈选响应
#[derive(Deserialize)]
#[allow(dead_code)] // 预留用于未来功能
pub struct LassoResponse {
    pub selected_edges: Vec<usize>,
    pub loops: Vec<Vec<[f64; 2]>>,
    pub connected_components: usize,
}

/// 缺口检测请求
#[derive(Serialize)]
pub struct GapDetectionRequest {
    pub tolerance: f64,
}

/// 缺口信息响应
#[derive(Deserialize, Clone)]
#[allow(dead_code)] // 预留用于未来功能
pub struct GapInfoResponse {
    pub id: usize,
    pub start: [f64; 2],
    pub end: [f64; 2],
    pub length: f64,
    pub gap_type: String,
}

/// 缺口检测响应
#[derive(Deserialize)]
#[allow(dead_code)] // 预留用于未来功能
pub struct GapDetectionResponse {
    pub gaps: Vec<GapInfoResponse>,
    pub total_count: usize,
}

/// 导出场景请求
#[derive(Serialize)]
#[allow(dead_code)] // 预留用于未来功能
pub struct ExportRequest<'a> {
    pub path: &'a str,
    pub format: &'a str,
}

/// 导出场景响应
#[derive(Deserialize)]
#[allow(dead_code)] // 预留用于未来功能
pub struct ExportResponse {
    pub success: bool,
    pub message: String,
}

impl ApiClient {
    pub fn new(base_url: &str) -> Self {
        // 去掉末尾的斜杠
        let base_url = base_url.trim_end_matches('/').to_string();
        Self {
            base_url,
            client: reqwest::Client::new(),
            ws_connection: Arc::new(Mutex::new(None)),
            ws_connected: Arc::new(Mutex::new(false)),
        }
    }

    /// 连接 WebSocket（P11 新增：实时交互）
    pub async fn connect_websocket(&self) -> Result<(), String> {
        // 如果已连接，跳过
        {
            let connected = self.ws_connected.lock().await;
            if *connected {
                return Ok(());
            }
        }

        // 将 http:// 替换为 ws://
        let ws_url = self.base_url
            .replace("http://", "ws://")
            .replace("https://", "wss://");
        
        let ws_endpoint = format!("{}/ws", ws_url);
        
        info!("正在连接 WebSocket: {}", ws_endpoint);

        match connect_async(&ws_endpoint).await {
            Ok((ws_stream, _)) => {
                let mut conn = self.ws_connection.lock().await;
                *conn = Some(ws_stream);
                
                let mut connected = self.ws_connected.lock().await;
                *connected = true;
                
                info!("WebSocket 连接成功");
                Ok(())
            }
            Err(e) => {
                info!("WebSocket 连接失败（后端可能未运行）: {}", e);
                Err(format!("WebSocket 连接失败：{}", e))
            }
        }
    }

    /// 断开 WebSocket 连接（P11 新增）
    pub async fn disconnect_websocket(&self) -> Result<(), String> {
        let mut conn = self.ws_connection.lock().await;
        *conn = None;
        
        let mut connected = self.ws_connected.lock().await;
        *connected = false;
        
        info!("WebSocket 已断开");
        Ok(())
    }

    /// 检查 WebSocket 是否已连接
    pub async fn is_websocket_connected(&self) -> bool {
        let connected = self.ws_connected.lock().await;
        *connected
    }

    /// 通过 WebSocket 发送边选择事件（P11 新增：实时交互）
    pub async fn ws_select_edge(&self, edge_id: usize) -> Result<WsTraceResult, String> {
        let mut conn_guard = self.ws_connection.lock().await;
        let conn_opt: &mut Option<WsStream> = &mut *conn_guard;
        
        if let Some(conn) = conn_opt {
            // 发送选择消息
            let select_msg = serde_json::json!({
                "type": "select_edge",
                "edge_id": edge_id
            });
            
            conn.send(Message::Text(select_msg.to_string())).await
                .map_err(|e| format!("发送消息失败：{}", e))?;
            
            // 等待响应（超时 5 秒）
            match tokio::time::timeout(
                std::time::Duration::from_secs(5),
                conn.next()
            ).await {
                Ok(Some(Ok(Message::Text(text)))) => {
                    // 解析响应
                    if let Ok(response) = serde_json::from_str::<WsMessage>(&text) {
                        match response {
                            WsMessage::TraceResult { edges, loop_closed } => {
                                Ok(WsTraceResult { edges, loop_closed })
                            }
                            WsMessage::Error { message } => {
                                Err(message)
                            }
                            _ => Err("意外的响应类型".to_string())
                        }
                    } else {
                        Err("解析响应失败".to_string())
                    }
                }
                Ok(Some(Err(e))) => Err(format!("接收错误：{}", e)),
                Ok(None) => Err("连接已关闭".to_string()),
                Err(_) => Err("响应超时（5 秒）".to_string()),
                _ => Err("未知错误".to_string())
            }
        } else {
            Err("WebSocket 未连接".to_string())
        }
    }

    /// 通过 WebSocket 发送缺口检测（P11 新增）
    pub async fn ws_detect_gaps(&self, tolerance: f64) -> Result<WsGapDetectionResult, String> {
        let mut conn_guard = self.ws_connection.lock().await;
        let conn_opt: &mut Option<WsStream> = &mut *conn_guard;
        
        if let Some(conn) = conn_opt {
            // 发送检测消息
            let detect_msg = serde_json::json!({
                "type": "detect_gaps",
                "tolerance": tolerance
            });
            
            conn.send(Message::Text(detect_msg.to_string())).await
                .map_err(|e| format!("发送消息失败：{}", e))?;
            
            // 等待响应（超时 10 秒）
            match tokio::time::timeout(
                std::time::Duration::from_secs(10),
                conn.next()
            ).await {
                Ok(Some(Ok(Message::Text(text)))) => {
                    // 解析响应
                    if let Ok(response) = serde_json::from_str::<WsMessage>(&text) {
                        match response {
                            WsMessage::GapsDetected { gaps } => {
                                Ok(WsGapDetectionResult { gaps })
                            }
                            WsMessage::Error { message } => {
                                Err(message)
                            }
                            _ => Err("意外的响应类型".to_string())
                        }
                    } else {
                        Err("解析响应失败".to_string())
                    }
                }
                Ok(Some(Err(e))) => Err(format!("接收错误：{}", e)),
                Ok(None) => Err("连接已关闭".to_string()),
                Err(_) => Err("响应超时（10 秒）".to_string()),
                _ => Err("未知错误".to_string())
            }
        } else {
            Err("WebSocket 未连接".to_string())
        }
    }

    /// 发送 WebSocket 心跳（P11 新增：保持连接）
    pub async fn ws_ping(&self) -> Result<(), String> {
        let mut conn_guard = self.ws_connection.lock().await;
        let conn_opt: &mut Option<WsStream> = &mut *conn_guard;
        
        if let Some(conn) = conn_opt {
            let ping_msg = serde_json::json!({
                "type": "ping"
            });
            
            conn.send(Message::Text(ping_msg.to_string())).await
                .map_err(|e| format!("发送心跳失败：{}", e))?;
            
            Ok(())
        } else {
            Err("WebSocket 未连接".to_string())
        }
    }

    /// 加载文件 - 调用后端 /process 接口（渐进式渲染）
    /// 
    /// # 渐进式渲染流程
    /// 
    /// 1. **阶段 1（快速）**：解析 DXF → 提取原始边 → 立即返回（~1 秒）
    /// 2. **阶段 2（后台）**：后端构建拓扑 → 完成后通过 WebSocket 推送更新
    pub async fn load_file(&mut self, path: &str) -> Result<Vec<Edge>, String> {
        let url = format!("{}/process", self.base_url);

        // 读取文件内容
        let file_content = tokio::fs::read(path)
            .await
            .map_err(|e| format!("读取文件失败：{}", e))?;

        let file_name = path.split('\\').next_back().or_else(|| path.split('/').next_back())
            .unwrap_or("unknown");

        let part = multipart::Part::bytes(file_content)
            .file_name(file_name.to_string());

        let form = multipart::Form::new()
            .part("file", part);

        // 渐进式渲染：阶段 1 只需 1-2 秒，超时设为 10 秒
        let response = tokio::time::timeout(
            std::time::Duration::from_secs(10),  // 10 秒超时（阶段 1 快速渲染）
            self.client.post(&url).multipart(form).send()
        )
        .await
        .map_err(|_| "请求超时（10 秒），文件可能无法解析")?
        .map_err(|e| format!("请求失败：{}", e))?;

        // 解析响应
        let result: orchestrator::api::ProcessResponse = response
            .json()
            .await
            .map_err(|e| format!("解析响应失败：{}", e))?;

        // 处理阶段 1 响应（快速渲染）
        match result.status {
            orchestrator::api::ProcessStatus::Completed => {
                info!("阶段 1 完成：{}", result.message);
                Ok(result.edges.unwrap_or_default())
            }
            orchestrator::api::ProcessStatus::Partial => {
                info!("阶段 1 完成（有警告）：{}", result.message);
                Ok(result.edges.unwrap_or_default())
            }
            orchestrator::api::ProcessStatus::Failed => {
                Err(result.message)
            }
        }
    }

    /// 自动追踪 - 调用后端 /interact/auto_trace 接口（带超时控制）
    pub async fn auto_trace(&mut self, edge_id: usize) -> Result<AutoTraceResponse, String> {
        let url = format!("{}/interact/auto_trace", self.base_url);

        let response = tokio::time::timeout(
            std::time::Duration::from_secs(10),
            self.client.post(&url).json(&AutoTraceRequest { edge_id }).send()
        )
        .await
        .map_err(|_| "自动追踪超时（10 秒）")?
        .map_err(|e| format!("请求失败：{}", e))?;

        let result: AutoTraceResponse = response
            .json()
            .await
            .map_err(|e| format!("解析响应失败：{}", e))?;

        Ok(result)
    }

    /// 圈选区域 - 调用后端 /interact/lasso 接口（带超时控制）
    #[allow(dead_code)] // 预留用于未来功能
    pub async fn lasso(&mut self, polygon: Vec<[f64; 2]>) -> Result<LassoResponse, String> {
        let url = format!("{}/interact/lasso", self.base_url);

        let response = tokio::time::timeout(
            std::time::Duration::from_secs(10),
            self.client.post(&url).json(&LassoRequest { polygon }).send()
        )
        .await
        .map_err(|_| "圈选超时（10 秒）")?
        .map_err(|e| format!("请求失败：{}", e))?;

        let result: LassoResponse = response
            .json()
            .await
            .map_err(|e| format!("解析响应失败：{}", e))?;

        Ok(result)
    }

    /// 缺口检测 - 调用后端 /interact/detect_gaps 接口（带超时控制）
    pub async fn detect_gaps(&mut self, tolerance: f64) -> Result<GapDetectionResponse, String> {
        let url = format!("{}/interact/detect_gaps", self.base_url);

        let response = tokio::time::timeout(
            std::time::Duration::from_secs(15),
            self.client.post(&url).json(&GapDetectionRequest { tolerance }).send()
        )
        .await
        .map_err(|_| "缺口检测超时（15 秒）")?
        .map_err(|e| format!("请求失败：{}", e))?;

        let result: GapDetectionResponse = response
            .json()
            .await
            .map_err(|e| format!("解析响应失败：{}", e))?;

        Ok(result)
    }

    /// 导出场景 - 调用后端 /export 接口
    pub async fn export_scene(&mut self, path: &str, edges: &[Edge], format: &str) -> Result<(), String> {
        let url = format!("{}/export", self.base_url);

        // 直接发送 JSON 对象，让后端解析
        let form = multipart::Form::new()
            .text("path", path.to_string())
            .text("format", format.to_string())
            .text("edges", serde_json::to_string(edges)
                .map_err(|e| format!("序列化边数据失败：{}", e))?);

        let response = self.client
            .post(&url)
            .multipart(form)
            .send()
            .await
            .map_err(|e| format!("请求失败：{}", e))?;

        let result: ExportResponse = response
            .json()
            .await
            .map_err(|e| format!("解析响应失败：{}", e))?;

        if result.success {
            info!("导出成功：{}", path);
            Ok(())
        } else {
            Err(result.message)
        }
    }

    /// 本地导出场景（不依赖后端）
    #[allow(dead_code)] // 预留用于未来功能
    pub async fn export_scene_local(&self, path: &str, edges: &[Edge], format: &str) -> Result<(), String> {
        // 根据格式序列化边数据
        let content: Vec<u8> = match format {
            "json" => serde_json::to_string_pretty(edges)
                .map_err(|e| format!("序列化 JSON 失败：{}", e))?
                .into_bytes(),
            "bincode" => bincode::serialize(edges)
                .map_err(|e| format!("序列化 bincode 失败：{}", e))?,
            _ => return Err(format!("不支持的导出格式：{}", format)),
        };

        // 写入文件
        let path_ref: &Path = Path::new(path);
        if let Some(parent) = path_ref.parent() {
            tokio::fs::create_dir_all(parent)
                .await
                .map_err(|e| format!("创建目录失败：{}", e))?;
        }

        tokio::fs::write(path_ref, content)
            .await
            .map_err(|e| format!("写入文件失败：{}", e))?;

        info!("本地导出成功：{}", path);
        Ok(())
    }
}

// ============================================================================
// WebSocket 消息类型（P11 新增：与后端 orchestrator 通信）
// ============================================================================

/// WebSocket 消息类型（从后端接收）
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum WsMessage {
    /// 连接确认
    #[serde(rename = "connected")]
    Connected { session_id: String },
    /// 边选择事件
    #[serde(rename = "edge_selected")]
    EdgeSelected { edge_id: usize },
    /// 追踪结果
    #[serde(rename = "trace_result")]
    TraceResult { edges: Vec<usize>, loop_closed: bool },
    /// 缺口检测结果
    #[serde(rename = "gaps_detected")]
    GapsDetected { gaps: Vec<WsGapInfoResponse> },
    /// 错误消息
    #[serde(rename = "error")]
    Error { message: String },
    /// 心跳响应
    #[serde(rename = "pong")]
    Pong,
}

/// 缺口信息响应（WebSocket）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WsGapInfoResponse {
    pub id: usize,
    pub start: [f64; 2],
    pub end: [f64; 2],
    pub length: f64,
    pub gap_type: String,
}

/// WebSocket 追踪结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WsTraceResult {
    pub edges: Vec<usize>,
    pub loop_closed: bool,
}

/// WebSocket 缺口检测结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WsGapDetectionResult {
    pub gaps: Vec<WsGapInfoResponse>,
}
