use super::{
    gap_info_to_response, ApiState, AutoTraceResponse, BoundarySemanticRequest,
    GapDetectionRequest, GapDetectionResponse, GapInfoResponse, InteractionStateResponse,
    LassoRequest, LassoResponse, SelectEdgeRequest, SnapBridgeRequest,
};
use axum::{extract::State, http::StatusCode, Json};
use common_types::Point2;
use interact::InteractService;

/// 交互 API - 选边追踪处理器
pub(super) async fn interact_auto_trace_handler(
    State(state): State<ApiState>,
    Json(request): Json<SelectEdgeRequest>,
) -> Result<Json<AutoTraceResponse>, StatusCode> {
    tracing::info!("收到选边追踪请求：edge_id={}", request.edge_id);

    let mut interact = state.interact.lock().await;

    match interact.auto_trace_from_edge(request.edge_id) {
        Ok(result) => {
            let loop_points = result
                .loop_
                .as_ref()
                .map(|l| l.points.iter().map(|p| [p[0], p[1]]).collect());

            Ok(Json(AutoTraceResponse {
                success: true,
                loop_points,
                message: format!(
                    "成功追踪到 {} 个点",
                    result.loop_.as_ref().map(|l| l.points.len()).unwrap_or(0)
                ),
            }))
        }
        Err(e) => {
            tracing::warn!("选边追踪失败：{:?}", e);
            Ok(Json(AutoTraceResponse {
                success: false,
                loop_points: None,
                message: format!("追踪失败：{:?}", e),
            }))
        }
    }
}

/// 交互 API - 圈选区域处理器
pub(super) async fn interact_lasso_handler(
    State(state): State<ApiState>,
    Json(request): Json<LassoRequest>,
) -> Result<Json<LassoResponse>, StatusCode> {
    tracing::info!("收到圈选请求，多边形点数={}", request.polygon.len());

    let polygon: Vec<Point2> = request.polygon.iter().map(|p| [p[0], p[1]]).collect();
    let mut interact = state.interact.lock().await;

    match interact.extract_from_lasso(&polygon) {
        Ok(result) => {
            let loops = result
                .loops
                .iter()
                .map(|l| l.points.iter().map(|p| [p[0], p[1]]).collect())
                .collect();

            Ok(Json(LassoResponse {
                selected_edges: result.selected_edges,
                loops,
                connected_components: result.connected_components,
            }))
        }
        Err(e) => {
            tracing::warn!("圈选失败：{:?}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

/// 交互 API - 缺口检测处理器
pub(super) async fn interact_detect_gaps_handler(
    State(state): State<ApiState>,
    Json(request): Json<GapDetectionRequest>,
) -> Result<Json<GapDetectionResponse>, StatusCode> {
    tracing::info!("收到缺口检测请求：tolerance={}", request.tolerance);

    let interact = state.interact.lock().await;

    match interact.detect_gaps(request.tolerance) {
        Ok(gaps) => {
            let gap_responses: Vec<GapInfoResponse> =
                gaps.iter().map(gap_info_to_response).collect();

            Ok(Json(GapDetectionResponse {
                gaps: gap_responses,
                total_count: gaps.len(),
            }))
        }
        Err(e) => {
            tracing::warn!("缺口检测失败：{:?}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

/// 交互 API - 缺口桥接处理器
pub(super) async fn interact_snap_bridge_handler(
    State(state): State<ApiState>,
    Json(request): Json<SnapBridgeRequest>,
) -> Result<StatusCode, StatusCode> {
    tracing::info!("收到缺口桥接请求：gap_id={}", request.gap_id);

    let mut interact = state.interact.lock().await;

    match interact.apply_snap_bridge(request.gap_id) {
        Ok(_) => Ok(StatusCode::OK),
        Err(e) => {
            tracing::warn!("缺口桥接失败：{:?}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

/// 交互 API - 边界语义设置处理器
pub(super) async fn interact_set_boundary_semantic_handler(
    State(state): State<ApiState>,
    Json(request): Json<BoundarySemanticRequest>,
) -> Result<StatusCode, StatusCode> {
    tracing::info!(
        "收到边界语义设置请求：segment_id={}, semantic={}",
        request.segment_id,
        request.semantic
    );

    use common_types::scene::BoundarySemantic;
    let semantic = match request.semantic.as_str() {
        "hard_wall" => BoundarySemantic::HardWall,
        "absorptive_wall" => BoundarySemantic::AbsorptiveWall,
        "door" => BoundarySemantic::Door,
        "window" => BoundarySemantic::Window,
        "opening" => BoundarySemantic::Opening,
        s => BoundarySemantic::Custom(s.to_string()),
    };

    let mut interact = state.interact.lock().await;

    match interact.set_boundary_semantic(request.segment_id, semantic) {
        Ok(_) => Ok(StatusCode::OK),
        Err(e) => {
            tracing::warn!("边界语义设置失败：{:?}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

/// 交互 API - 状态查询处理器
pub(super) async fn interact_state_handler(
    State(state): State<ApiState>,
) -> Result<Json<InteractionStateResponse>, StatusCode> {
    let interact = state.interact.lock().await;

    let state_ref = interact.get_state();
    let selected_edges: Vec<usize> = state_ref.selected_edges.iter().copied().collect();
    let detected_gaps: Vec<GapInfoResponse> = state_ref
        .detected_gaps
        .iter()
        .map(gap_info_to_response)
        .collect();

    Ok(Json(InteractionStateResponse {
        total_edges: state_ref.edges.len(),
        selected_edges,
        detected_gaps,
    }))
}
