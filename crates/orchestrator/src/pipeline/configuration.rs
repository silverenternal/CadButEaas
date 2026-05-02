use export::service::ExportConfig as ExportServiceConfig;
use parser::ParserService;
use topo::service::TopoConfig as TopoServiceConfig;
use validator::service::ValidatorConfig as ValidatorServiceConfig;
use vectorize::{RasterStrategy, VectorizeConfig};

/// 创建 ParserService 并应用 DXF 配置
pub(super) fn create_parser_service(parser_config: &config::ParserConfig) -> ParserService {
    ParserService::new().with_dxf_filter(
        parser_config.dxf.ignore_text,
        parser_config.dxf.ignore_dimensions,
        parser_config.dxf.ignore_hatch,
    )
}

/// 转换矢量化配置
pub(super) fn vectorize_config(config: &config::CadConfig) -> VectorizeConfig {
    let parser_config = &config.parser;
    let raster_strategy = config
        .raster
        .strategy
        .parse::<RasterStrategy>()
        .unwrap_or(RasterStrategy::Auto);
    VectorizeConfig {
        threshold: parser_config.pdf.threshold,
        snap_tolerance_px: parser_config.pdf.vectorize_tolerance_px,
        min_line_length_px: parser_config.pdf.min_line_length_px,
        max_angle_dev_deg: 5.0,
        skeletonize: true,
        #[cfg(feature = "opencv")]
        use_opencv: true,
        #[cfg(not(feature = "opencv"))]
        use_opencv: false,
        adaptive_threshold: true,
        use_hough: false,
        preprocessing: vectorize::config::PreprocessingConfig::default(),
        line_type_detection: false,
        arc_fitting: false,
        gap_filling: false,
        quality_assessment: false,
        text_separation: false,
        dpi_adaptive: true,
        reference_dpi: 300.0,
        dpi_scale_factor: 1.0,
        opencv_approx_epsilon: Some(2.0),
        max_pixels: 30_000_000,
        use_accelerator_edge_detect: true,
        auto_crop_paper: true,
        perspective_correction: true,
        hough_gap_filling: true,
        hough_threshold: 50,
        architectural_correction: true,
        adaptive_params: true,
        raster_strategy,
        max_retries: config.raster.max_retries.max(1),
        vlm_backend: vectorize::RasterVlmBackendConfig::default(),
    }
}

/// 转换拓扑配置
pub(super) fn topo_config(config: &config::CadConfig) -> TopoServiceConfig {
    use topo::service::TopoAlgorithm;

    let algorithm = match config.topology.algorithm.as_str() {
        "halfedge" => TopoAlgorithm::Halfedge,
        _ => TopoAlgorithm::Dfs,
    };

    TopoServiceConfig {
        tolerance: common_types::geometry::ToleranceConfig {
            snap_tolerance: config.topology.snap_tolerance_mm,
            min_line_length: config.topology.min_line_length_mm,
            max_angle_deviation: config.topology.merge_angle_tolerance_deg,
            units: Some(common_types::LengthUnit::Mm),
        },
        layer_filter: None,
        algorithm,
        skip_intersection_check: config.topology.skip_intersection_check,
        enable_parallel: config.topology.enable_parallel,
        parallel_threshold: config.topology.parallel_threshold,
    }
}

/// 转换验证器配置
pub(super) fn validator_config(config: &config::CadConfig) -> ValidatorServiceConfig {
    ValidatorServiceConfig {
        closure_tolerance: config.validator.closure_tolerance_mm,
        min_edge_length: config.validator.min_edge_length_mm,
        min_angle_degrees: config.validator.min_angle_deg,
    }
}

/// 转换导出配置
pub(super) fn export_config(config: &config::CadConfig) -> ExportServiceConfig {
    ExportServiceConfig {
        format: match config.export.format.as_str() {
            "json" => export::formats::ExportFormat::Json,
            "bincode" | "binary" => export::formats::ExportFormat::Binary,
            _ => export::formats::ExportFormat::Json,
        },
        pretty_json: config.export.json_indent > 0,
        target_units: None,
    }
}
