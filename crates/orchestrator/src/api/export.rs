use super::{uuid_simple, ApiState, ExportRequest, ExportResponse};
use axum::{extract::State, http::StatusCode, Json};

/// 导出处理器
pub(super) async fn export_handler(
    State(state): State<ApiState>,
    Json(request): Json<ExportRequest>,
) -> Result<Json<ExportResponse>, StatusCode> {
    use export::formats::ExportFormat;

    tracing::info!("收到导出请求：format={}", request.format);

    let interact = state.interact.lock().await;
    let scene_state = interact.get_scene_state();
    drop(interact);

    let format = match request.format.to_lowercase().as_str() {
        "json" => ExportFormat::Json,
        "bincode" | "binary" => ExportFormat::Binary,
        "dxf" => {
            return Ok(Json(ExportResponse {
                success: false,
                message: "DXF 导出暂不支持".to_string(),
                download_url: None,
                file_name: None,
                file_size: 0,
            }));
        }
        _ => {
            return Err(StatusCode::BAD_REQUEST);
        }
    };

    let export_service = state.pipeline.export();

    match export_service.export(&scene_state) {
        Ok(export_result) => {
            let file_name = format!(
                "cad_export_{}.{}",
                uuid_simple(),
                match format {
                    ExportFormat::Json => "json",
                    ExportFormat::Binary => "bin",
                }
            );

            let temp_path = std::env::temp_dir().join(&file_name);

            if let Err(e) = std::fs::write(&temp_path, &export_result.bytes) {
                tracing::error!("写入临时文件失败：{}", e);
                return Err(StatusCode::INTERNAL_SERVER_ERROR);
            }

            tracing::info!(
                "导出成功：file_name={}, size={} bytes",
                file_name,
                export_result.bytes.len()
            );

            Ok(Json(ExportResponse {
                success: true,
                message: "导出成功".to_string(),
                download_url: Some(format!("/download/{}", file_name)),
                file_name: Some(file_name),
                file_size: export_result.bytes.len(),
            }))
        }
        Err(e) => {
            tracing::error!("导出失败：{}", e);
            Ok(Json(ExportResponse {
                success: false,
                message: format!("导出失败：{}", e),
                download_url: None,
                file_name: None,
                file_size: 0,
            }))
        }
    }
}

/// 下载处理器
pub(super) async fn download_handler(
    State(_state): State<ApiState>,
    axum::extract::Path(filename): axum::extract::Path<String>,
) -> Result<axum::response::Response, StatusCode> {
    use axum::http::header;

    tracing::info!("收到下载请求：filename={}", filename);

    let temp_path = std::env::temp_dir().join(&filename);

    if !temp_path.exists() {
        tracing::warn!("文件不存在：{}", filename);
        return Err(StatusCode::NOT_FOUND);
    }

    let file_content = match std::fs::read(&temp_path) {
        Ok(content) => content,
        Err(e) => {
            tracing::error!("读取文件失败：{}", e);
            return Err(StatusCode::INTERNAL_SERVER_ERROR);
        }
    };

    let content_type = if filename.ends_with(".json") {
        "application/json"
    } else {
        "application/octet-stream"
    };

    let mut response = axum::response::Response::new(file_content.into());
    response
        .headers_mut()
        .insert(header::CONTENT_TYPE, content_type.parse().unwrap());
    response.headers_mut().insert(
        header::CONTENT_DISPOSITION,
        format!("attachment; filename=\"{}\"", filename)
            .parse()
            .unwrap(),
    );

    Ok(response)
}
