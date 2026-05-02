use image::GrayImage;
use std::time::Duration;
use vectorize::semantic::vlm_backend::{
    build_vlm_input, DisabledRasterVlmBackend, HeuristicRasterVlmBackend, HttpRasterVlmBackend,
    RasterVlmBackend,
};
use vectorize::{RasterSemanticExtractor, RasterVlmBackendConfig, RasterVlmBackendKind};

fn sample_image() -> GrayImage {
    GrayImage::from_pixel(32, 32, image::Luma([255]))
}

fn sample_polylines() -> Vec<common_types::Polyline> {
    vec![vec![[0.0, 0.0], [100.0, 0.0]], vec![[2.0, 2.0], [20.0, 2.0]]]
}

#[test]
fn raster_vlm_backend_default_is_heuristic_with_http_endpoint() {
    let config = RasterVlmBackendConfig::default();

    assert_eq!(config.kind, RasterVlmBackendKind::Heuristic);
    assert_eq!(
        config.endpoint.as_deref(),
        Some("http://127.0.0.1:8765/analyze_raster")
    );
    assert!(config.timeout_ms > 0);
    assert!(config.max_thumbnail_px > 0);
}

#[test]
fn heuristic_and_disabled_backends_share_schema_contract() {
    let image = sample_image();
    let input = build_vlm_input(&image, &sample_polylines(), &[], &[], 64).unwrap();

    let disabled = DisabledRasterVlmBackend.analyze(&input).unwrap();
    assert_eq!(disabled.schema_version, "raster-vlm-1.0");
    assert_eq!(disabled.model_info.backend, "disabled");
    assert!(disabled.semantic_candidates.is_empty());

    let heuristic = HeuristicRasterVlmBackend.analyze(&input).unwrap();
    assert_eq!(heuristic.schema_version, "raster-vlm-1.0");
    assert_eq!(heuristic.model_info.backend, "heuristic");
    assert!(!heuristic.semantic_candidates.is_empty());
}

#[test]
fn http_backend_failure_falls_back_to_heuristic_extractor() {
    let extractor = RasterSemanticExtractor::with_vlm_backend(RasterVlmBackendConfig {
        kind: RasterVlmBackendKind::Http,
        endpoint: Some("http://127.0.0.1:9/analyze_raster".to_string()),
        timeout_ms: 50,
        max_thumbnail_px: 64,
    });

    let result = extractor.extract(&sample_image(), &sample_polylines());

    assert_eq!(result.vlm_backend, "heuristic");
    assert!(result.vlm_fallback_reason.is_some());
    assert!(!result.semantic_candidates.is_empty());
}

#[test]
fn http_backend_rejects_non_http_endpoint_for_auditable_failure() {
    let image = sample_image();
    let input = build_vlm_input(&image, &sample_polylines(), &[], &[], 0).unwrap();
    let backend = HttpRasterVlmBackend::new("https://127.0.0.1/analyze_raster", Duration::from_millis(50));

    let err = backend.analyze(&input).unwrap_err().to_string();
    assert!(err.contains("HTTP endpoint must start with http://"));
}
