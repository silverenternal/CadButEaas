use super::RasterProcessingOptions;
use common_types::{Polyline, RasterSceneMetadata, ScaleConfidence, SourceImageMetadata};
use vectorize::{RasterVectorizationReport, SemanticCandidate};

#[derive(Debug, Clone)]
pub(super) struct RasterCoordinateTransform {
    pub(super) px_to_mm: Option<[f64; 2]>,
    pub(super) confidence: ScaleConfidence,
    pub(super) source: Option<String>,
}

pub(super) fn distance_2d(a: [f64; 2], b: [f64; 2]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    (dx * dx + dy * dy).sqrt()
}

pub(super) fn raster_transform(
    info: Option<&raster_loader::RasterImageInfo>,
    options: &RasterProcessingOptions,
) -> RasterCoordinateTransform {
    if let Some(calibration) = &options.scale_calibration {
        let distance_px = calibration
            .points_px
            .map(|(a, b)| distance_2d(a, b))
            .unwrap_or(calibration.known_distance_px);
        if distance_px > 0.0 && calibration.known_distance_mm > 0.0 {
            let scale = calibration.known_distance_mm / distance_px;
            return RasterCoordinateTransform {
                px_to_mm: Some([scale, scale]),
                confidence: ScaleConfidence::High,
                source: Some("user_calibration".to_string()),
            };
        }
    }

    if let Some((dpi_x, dpi_y)) = options.dpi_override {
        if dpi_x > 0.0 && dpi_y > 0.0 {
            return RasterCoordinateTransform {
                px_to_mm: Some([25.4 / dpi_x, 25.4 / dpi_y]),
                confidence: ScaleConfidence::High,
                source: Some("dpi_override".to_string()),
            };
        }
    }

    if let Some(info) = info {
        if let (Some(dpi_x), Some(dpi_y)) = (info.dpi_x, info.dpi_y) {
            if dpi_x > 0.0 && dpi_y > 0.0 && info.dpi_trusted {
                return RasterCoordinateTransform {
                    px_to_mm: Some([25.4 / dpi_x, 25.4 / dpi_y]),
                    confidence: ScaleConfidence::Medium,
                    source: info.dpi_source.clone(),
                };
            }
        }
    }

    RasterCoordinateTransform {
        px_to_mm: None,
        confidence: ScaleConfidence::Unknown,
        source: None,
    }
}

pub(super) fn transform_polylines(
    polylines: &[Polyline],
    transform: &RasterCoordinateTransform,
) -> Vec<Polyline> {
    let Some([sx, sy]) = transform.px_to_mm else {
        return polylines.to_vec();
    };

    polylines
        .iter()
        .map(|polyline| {
            polyline
                .iter()
                .map(|point| [point[0] * sx, point[1] * sy])
                .collect()
        })
        .collect()
}

pub(super) fn raster_scene_metadata(
    info: Option<&raster_loader::RasterImageInfo>,
    transform: &RasterCoordinateTransform,
) -> RasterSceneMetadata {
    let dpi_from_transform = transform.px_to_mm.and_then(|[sx, sy]| {
        if transform.source.as_deref() == Some("dpi_override") && sx > 0.0 && sy > 0.0 {
            Some((25.4 / sx, 25.4 / sy))
        } else {
            None
        }
    });
    RasterSceneMetadata {
        source_image: info.map(|info| SourceImageMetadata {
            width_px: info.width,
            height_px: info.height,
            format: format!("{:?}", info.format),
            path: info.source_path.clone(),
        }),
        dpi: dpi_from_transform.or_else(|| info.and_then(|info| info.dpi_x.zip(info.dpi_y))),
        px_to_mm: transform.px_to_mm,
        scale_confidence: transform.confidence,
        calibration_source: transform.source.clone(),
    }
}

pub(super) fn dimension_summary_from_text_candidates(
    report: &RasterVectorizationReport,
) -> common_types::DimensionSummary {
    let mut summary = common_types::DimensionSummary::default();
    for candidate in &report.dimension_candidates {
        if let Some(value) = candidate.nominal_value {
            summary.linear_count += 1;
            summary.total_count += 1;
            summary.max_measurement = Some(
                summary
                    .max_measurement
                    .map_or(value, |existing| existing.max(value)),
            );
            summary.min_measurement = Some(
                summary
                    .min_measurement
                    .map_or(value, |existing| existing.min(value)),
            );
        }
    }

    if summary.total_count > 0 {
        return summary;
    }

    for candidate in &report.text_candidates {
        let normalized = candidate
            .content
            .trim()
            .replace(',', "")
            .replace("mm", "")
            .replace("MM", "");
        if let Ok(value) = normalized.parse::<f64>() {
            summary.linear_count += 1;
            summary.total_count += 1;
            summary.max_measurement = Some(
                summary
                    .max_measurement
                    .map_or(value, |existing| existing.max(value)),
            );
            summary.min_measurement = Some(
                summary
                    .min_measurement
                    .map_or(value, |existing| existing.min(value)),
            );
        }
    }
    summary
}

pub(super) fn vector_graph_semantic_candidates(polylines: &[Polyline]) -> Vec<SemanticCandidate> {
    let graph = vector_graph::CadGraph::from_polylines(polylines, 2.0);
    let node_count = graph.node_count().max(1) as f64;
    let edge_count = graph.edge_count().max(1) as f64;
    let connectivity = (edge_count / node_count).clamp(0.0, 2.0) / 2.0;

    polylines
        .iter()
        .enumerate()
        .take(256)
        .map(|(idx, polyline)| {
            let length = polyline
                .windows(2)
                .map(|segment| distance_2d(segment[0], segment[1]))
                .sum::<f64>();
            let semantic_type = if length > 120.0 {
                "hard_wall"
            } else if length > 40.0 {
                "opening"
            } else {
                "detail_line"
            };
            SemanticCandidate {
                target_id: idx,
                semantic_type: semantic_type.to_string(),
                confidence: (0.45 + connectivity * 0.35).clamp(0.0, 0.85),
                source: "vector_graph_rule".to_string(),
            }
        })
        .collect()
}
