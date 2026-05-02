use super::{gap_info_to_response, ApiState, GapInfoResponse, WsMessage};
use axum::{
    extract::{
        ws::{Message, WebSocket},
        State, WebSocketUpgrade,
    },
    response::IntoResponse,
};
use futures::{sink::SinkExt, stream::StreamExt};
use interact::InteractService;

/// WebSocket 处理器
pub(super) async fn websocket_handler(
    ws: WebSocketUpgrade,
    State(state): State<ApiState>,
) -> impl IntoResponse {
    let session_id = uuid::Uuid::new_v4().to_string();
    tracing::info!("WebSocket 连接：session_id={}", session_id);

    ws.on_upgrade(move |socket| handle_websocket(socket, state, session_id))
}

/// 处理 WebSocket 连接
async fn handle_websocket(socket: WebSocket, state: ApiState, session_id: String) {
    let (mut sender, mut receiver) = socket.split();

    let connected_msg = WsMessage::Connected {
        session_id: session_id.clone(),
    };
    if let Ok(json) = serde_json::to_string(&connected_msg) {
        let _ = sender.send(Message::Text(json)).await;
    }

    let mut topology_pushed = false;

    while let Some(msg) = receiver.next().await {
        if !topology_pushed {
            let interact = state.interact.lock().await;
            if interact.get_state().topology_ready {
                let edge_count = interact.get_state().edges.len();
                drop(interact);

                let ready_msg = WsMessage::TopologyReady { edge_count };
                if let Ok(json) = serde_json::to_string(&ready_msg) {
                    if sender.send(Message::Text(json)).await.is_err() {
                        break;
                    }
                }
                topology_pushed = true;
            }
        }

        match msg {
            Ok(Message::Text(text)) => {
                handle_text_message(&state, &mut sender, &text).await;
            }
            Ok(Message::Close(_)) => {
                tracing::info!("WebSocket 断开：session_id={}", session_id);
                break;
            }
            Err(e) => {
                tracing::error!("WebSocket 错误：{:?}", e);
                break;
            }
            _ => {}
        }
    }

    tracing::info!("WebSocket 连接结束：session_id={}", session_id);
}

async fn handle_text_message(
    state: &ApiState,
    sender: &mut futures::stream::SplitSink<WebSocket, Message>,
    text: &str,
) {
    let Ok(client_msg) = serde_json::from_str::<serde_json::Value>(text) else {
        return;
    };
    let Some(msg_type) = client_msg.get("type").and_then(|t| t.as_str()) else {
        return;
    };

    match msg_type {
        "select_edge" => {
            if let Some(edge_id) = client_msg.get("edge_id").and_then(|id| id.as_u64()) {
                send_trace_result(state, sender, edge_id as usize).await;
            }
        }
        "detect_gaps" => {
            let tolerance = client_msg
                .get("tolerance")
                .and_then(|t| t.as_f64())
                .unwrap_or(0.5);
            send_gap_result(state, sender, tolerance).await;
        }
        "ping" => {
            send_ws_message(sender, &WsMessage::Pong).await;
        }
        _ => {
            tracing::warn!("未知 WebSocket 消息类型：{}", msg_type);
        }
    }
}

async fn send_trace_result(
    state: &ApiState,
    sender: &mut futures::stream::SplitSink<WebSocket, Message>,
    edge_id: usize,
) {
    let mut interact = state.interact.lock().await;
    match interact.auto_trace_from_edge(edge_id) {
        Ok(trace_result) => {
            let edges = trace_result.path;
            let loop_closed = trace_result.loop_.is_some();
            send_ws_message(sender, &WsMessage::TraceResult { edges, loop_closed }).await;
        }
        Err(e) => {
            send_ws_message(
                sender,
                &WsMessage::Error {
                    message: e.to_string(),
                },
            )
            .await;
        }
    }
}

async fn send_gap_result(
    state: &ApiState,
    sender: &mut futures::stream::SplitSink<WebSocket, Message>,
    tolerance: f64,
) {
    let interact = state.interact.lock().await;
    match interact.detect_gaps(tolerance) {
        Ok(gaps) => {
            let gap_responses: Vec<GapInfoResponse> =
                gaps.iter().map(gap_info_to_response).collect();
            send_ws_message(
                sender,
                &WsMessage::GapsDetected {
                    gaps: gap_responses,
                },
            )
            .await;
        }
        Err(e) => {
            send_ws_message(
                sender,
                &WsMessage::Error {
                    message: e.to_string(),
                },
            )
            .await;
        }
    }
}

async fn send_ws_message(
    sender: &mut futures::stream::SplitSink<WebSocket, Message>,
    message: &WsMessage,
) {
    if let Ok(json) = serde_json::to_string(message) {
        let _ = sender.send(Message::Text(json)).await;
    }
}
