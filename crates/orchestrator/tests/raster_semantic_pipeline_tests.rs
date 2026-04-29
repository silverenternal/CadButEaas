use orchestrator::pipeline::{RasterProcessingOptions, ScaleCalibration};
use orchestrator::ProcessingPipeline;
use std::io::Cursor;
use vectorize::benchmark_data::{generate_case, RasterDegradation};
use vectorize::RasterStrategy;

fn png_bytes(degradation: RasterDegradation) -> Vec<u8> {
    let case = generate_case(degradation);
    let mut cursor = Cursor::new(Vec::new());
    case.image
        .write_to(&mut cursor, image::ImageFormat::Png)
        .unwrap();
    cursor.into_inner()
}

#[tokio::test]
async fn raster_semantic_pipeline_returns_report_scale_and_candidates() {
    let pipeline = ProcessingPipeline::new();
    let png = png_bytes(RasterDegradation::CleanLineArt);
    let result = pipeline
        .process_raster_bytes_with_options(
            &png,
            Some("plan.png"),
            RasterProcessingOptions {
                strategy: Some(RasterStrategy::CleanLineArt),
                dpi_override: Some((300.0, 300.0)),
                scale_calibration: None,
                debug_artifacts: false,
                semantic_mode: Some("semantic".to_string()),
                ocr_backend: Some("heuristic".to_string()),
                max_retries: Some(2),
            },
        )
        .await
        .unwrap();

    let report = result.raster_report.as_ref().unwrap();
    assert_eq!(report.schema_version, "raster-report-1.0");
    assert!(report.final_polyline_count > 0);
    assert!(!report.stage_stats.is_empty());
    assert!(!report.attempts.is_empty());
    assert!(!result.scene.edges.is_empty());
    assert!(!result.semantic_candidates.is_empty());

    let metadata = result.scene.raster_metadata.as_ref().unwrap();
    assert_eq!(metadata.dpi, Some((300.0, 300.0)));
    let scale = metadata.px_to_mm.unwrap();
    assert!((scale[0] - 25.4 / 300.0).abs() < 1e-9);
}

#[tokio::test]
async fn raster_scale_calibration_overrides_dpi() {
    let pipeline = ProcessingPipeline::new();
    let png = png_bytes(RasterDegradation::CleanLineArt);
    let result = pipeline
        .process_raster_bytes_with_options(
            &png,
            Some("plan.png"),
            RasterProcessingOptions {
                strategy: Some(RasterStrategy::CleanLineArt),
                dpi_override: Some((300.0, 300.0)),
                scale_calibration: Some(ScaleCalibration {
                    known_distance_px: 100.0,
                    known_distance_mm: 250.0,
                    points_px: None,
                }),
                debug_artifacts: false,
                semantic_mode: None,
                ocr_backend: None,
                max_retries: Some(1),
            },
        )
        .await
        .unwrap();

    let metadata = result.scene.raster_metadata.unwrap();
    assert_eq!(
        metadata.calibration_source.as_deref(),
        Some("user_calibration")
    );
    assert_eq!(metadata.px_to_mm, Some([2.5, 2.5]));
}
