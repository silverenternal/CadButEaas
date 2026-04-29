use accelerator_cpu::CpuAccelerator;
use vectorize::benchmark_data::{geometry_score, small_ci_set, RasterDegradation};
use vectorize::{RasterStrategy, VectorizeConfig, VectorizeService};

#[test]
fn raster_golden_cases_return_report_and_metrics() {
    let service =
        VectorizeService::new(Box::new(CpuAccelerator::new()), VectorizeConfig::default());

    for case in small_ci_set() {
        let config = VectorizeConfig {
            raster_strategy: match case.degradation {
                RasterDegradation::CleanLineArt => RasterStrategy::CleanLineArt,
                RasterDegradation::ScanNoise => RasterStrategy::ScannedPlan,
                RasterDegradation::JpegCompression => RasterStrategy::ScannedPlan,
                RasterDegradation::PhotoPerspective => RasterStrategy::PhotoPerspective,
                RasterDegradation::LowContrast => RasterStrategy::LowContrast,
                RasterDegradation::HandSketch => RasterStrategy::HandSketch,
            },
            max_retries: 2,
            ..Default::default()
        };

        let output = service
            .vectorize_image_detailed(&case.image, &config, false)
            .unwrap_or_else(|err| panic!("{} failed: {}", case.name, err));

        assert!(
            output.report.quality_score >= 0.0,
            "{} should carry quality score",
            case.name
        );
        assert!(
            output.report.stage_stats.len() >= 5,
            "{} should carry stage timings",
            case.name
        );
        assert!(
            !output.report.attempts.is_empty(),
            "{} should record attempts",
            case.name
        );

        let (precision, recall) = geometry_score(&output.polylines, &case.ground_truth);
        assert!(
            precision >= 0.10,
            "{} precision too low: {:.3}",
            case.name,
            precision
        );
        assert!(
            recall >= 0.10,
            "{} recall too low: {:.3}",
            case.name,
            recall
        );
    }
}

#[test]
fn debug_artifacts_are_opt_in() {
    let service = VectorizeService::with_default();
    let case = small_ci_set()
        .into_iter()
        .find(|case| case.degradation == RasterDegradation::CleanLineArt)
        .unwrap();

    let no_debug = service
        .vectorize_image_detailed(&case.image, &VectorizeConfig::default(), false)
        .unwrap();
    assert!(no_debug.report.debug_artifacts.is_empty());

    let debug = service
        .vectorize_image_detailed(&case.image, &VectorizeConfig::default(), true)
        .unwrap();
    assert!(!debug.report.debug_artifacts.is_empty());
    assert!(debug
        .report
        .debug_artifacts
        .iter()
        .any(|artifact| artifact.name == "binary"));
}
