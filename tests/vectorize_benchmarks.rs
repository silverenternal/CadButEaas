//! Workspace-level vectorize performance budget smoke tests.

use std::time::Instant;
use vectorize::benchmark_data::{generate_case, RasterDegradation};
use vectorize::{RasterStrategy, VectorizeConfig, VectorizeService};

#[test]
fn vectorize_small_ci_budget_reports_stage_timings() {
    let service = VectorizeService::with_default();
    let cases = [
        (RasterDegradation::CleanLineArt, RasterStrategy::CleanLineArt),
        (RasterDegradation::ScanNoise, RasterStrategy::ScannedPlan),
        (RasterDegradation::LowContrast, RasterStrategy::LowContrast),
    ];

    for (degradation, strategy) in cases {
        let case = generate_case(degradation);
        let config = VectorizeConfig {
            raster_strategy: strategy,
            max_retries: 2,
            ..Default::default()
        };

        let start = Instant::now();
        let output = service
            .vectorize_image_detailed(&case.image, &config, false)
            .unwrap();
        let elapsed = start.elapsed();

        assert!(
            elapsed.as_millis() < 2_000,
            "{} exceeded CI budget: {:?}",
            case.name,
            elapsed
        );
        assert!(
            output.report.stage_stats.iter().all(|s| s.duration_ms < 2_000),
            "{} stage budget exceeded: {:?}",
            case.name,
            output.report.stage_stats
        );
        assert!(
            !output.report.recommendations.is_empty(),
            "{} should include degradation advice or success guidance",
            case.name
        );
    }
}

#[test]
#[ignore]
fn vectorize_large_local_budget() {
    let service = VectorizeService::with_default();
    let case = generate_case(RasterDegradation::PhotoPerspective);
    let config = VectorizeConfig {
        raster_strategy: RasterStrategy::PhotoPerspective,
        max_retries: 3,
        ..Default::default()
    };

    let start = Instant::now();
    let output = service
        .vectorize_image_detailed(&case.image, &config, true)
        .unwrap();
    let elapsed = start.elapsed();

    println!(
        "large local vectorize: {:?}, polylines={}, stages={:?}",
        elapsed,
        output.polylines.len(),
        output.report.stage_stats
    );
    assert!(elapsed.as_secs() < 10);
}
