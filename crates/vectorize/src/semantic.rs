//! Thin raster semantic extraction facade.
//!
//! This module wires together the current heuristic OCR, VLM-style dimension
//! parsing, CAD symbol matching, primitive fitting, and rule-based geometry
//! semantics behind one stable entry point.

pub mod schema;
pub mod vlm_backend;

use common_types::Polyline;
use image::GrayImage;

use crate::algorithms::{
    fit_best_primitive, FitData, HeuristicOcrBackend, OcrBackend, SymbolClassifier,
};
use crate::pipeline::{
    DimensionCandidate, PrimitiveCandidate, SemanticCandidate, SymbolCandidate, TextCandidate,
};
use vlm_backend::{
    build_vlm_input, DisabledRasterVlmBackend, HeuristicRasterVlmBackend, HttpRasterVlmBackend,
    RasterVlmBackend, RasterVlmBackendConfig, RasterVlmBackendKind,
};

#[derive(Debug, Clone, Default)]
pub struct RasterSemanticExtraction {
    pub primitive_candidates: Vec<PrimitiveCandidate>,
    pub text_candidates: Vec<TextCandidate>,
    pub symbol_candidates: Vec<SymbolCandidate>,
    pub dimension_candidates: Vec<DimensionCandidate>,
    pub semantic_candidates: Vec<SemanticCandidate>,
    pub vlm_backend: String,
    pub vlm_model_name: Option<String>,
    pub vlm_latency_ms: Option<u64>,
    pub vlm_fallback_reason: Option<String>,
    pub vlm_warnings: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct RasterSemanticExtractor {
    max_polylines: usize,
    min_text_confidence: f64,
    vlm: RasterVlmBackendConfig,
}

impl Default for RasterSemanticExtractor {
    fn default() -> Self {
        Self {
            max_polylines: 256,
            min_text_confidence: 0.5,
            vlm: RasterVlmBackendConfig::default(),
        }
    }
}

impl RasterSemanticExtractor {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_vlm_backend(vlm: RasterVlmBackendConfig) -> Self {
        Self {
            vlm,
            ..Self::default()
        }
    }

    pub fn extract(&self, image: &GrayImage, polylines: &[Polyline]) -> RasterSemanticExtraction {
        let primitive_candidates = self.primitive_candidates(polylines);
        let text_candidates = self.text_candidates(image);
        let symbol_candidates = self.symbol_candidates(image);
        let mut vlm_fallback_reason = None;

        let vlm_output = match self.run_configured_backend(
            image,
            polylines,
            &text_candidates,
            &symbol_candidates,
        ) {
            Ok(output) => output,
            Err(err) => {
                vlm_fallback_reason = Some(err.to_string());
                self.run_heuristic_backend(image, polylines, &text_candidates, &symbol_candidates)
            }
        };

        let mut symbol_candidates = symbol_candidates;
        symbol_candidates.extend(vlm_output.symbol_candidates);
        let dimension_candidates = fuse_dimensions(vlm_output.dimension_candidates);
        let semantic_candidates = fuse_semantics(vlm_output.semantic_candidates);

        RasterSemanticExtraction {
            primitive_candidates,
            text_candidates,
            symbol_candidates,
            dimension_candidates,
            semantic_candidates,
            vlm_backend: vlm_output.model_info.backend,
            vlm_model_name: vlm_output.model_info.model_name,
            vlm_latency_ms: vlm_output.model_info.latency_ms,
            vlm_fallback_reason,
            vlm_warnings: vlm_output.warnings,
        }
    }

    fn run_configured_backend(
        &self,
        image: &GrayImage,
        polylines: &[Polyline],
        text_candidates: &[TextCandidate],
        symbol_candidates: &[SymbolCandidate],
    ) -> Result<schema::RasterVlmOutput, vlm_backend::RasterVlmError> {
        let input = build_vlm_input(
            image,
            polylines,
            text_candidates,
            symbol_candidates,
            self.vlm.max_thumbnail_px,
        )?;
        match self.vlm.kind {
            RasterVlmBackendKind::Disabled => DisabledRasterVlmBackend.analyze(&input),
            RasterVlmBackendKind::Heuristic => HeuristicRasterVlmBackend.analyze(&input),
            RasterVlmBackendKind::Http => {
                let endpoint = self
                    .vlm
                    .endpoint
                    .clone()
                    .unwrap_or_else(|| RasterVlmBackendConfig::default().endpoint.unwrap());
                let backend = HttpRasterVlmBackend::new(
                    endpoint,
                    std::time::Duration::from_millis(self.vlm.timeout_ms),
                );
                backend.analyze(&input)
            }
        }
    }

    fn run_heuristic_backend(
        &self,
        image: &GrayImage,
        polylines: &[Polyline],
        text_candidates: &[TextCandidate],
        symbol_candidates: &[SymbolCandidate],
    ) -> schema::RasterVlmOutput {
        let input = build_vlm_input(image, polylines, text_candidates, symbol_candidates, 0)
            .expect("thumbnail disabled avoids image encoding failure");
        HeuristicRasterVlmBackend
            .analyze(&input)
            .expect("heuristic VLM backend is infallible")
    }

    fn primitive_candidates(&self, polylines: &[Polyline]) -> Vec<PrimitiveCandidate> {
        polylines
            .iter()
            .take(self.max_polylines)
            .filter_map(|polyline| {
                fit_best_primitive(polyline, 2.0).map(|fit| {
                    let (primitive_type, start, end) = match &fit.data {
                        FitData::Line(line) => ("line", line.start, line.end),
                        FitData::Arc(arc) => {
                            let start = [
                                arc.center[0] + arc.radius * arc.start_angle.cos(),
                                arc.center[1] + arc.radius * arc.start_angle.sin(),
                            ];
                            let end = [
                                arc.center[0] + arc.radius * arc.end_angle.cos(),
                                arc.center[1] + arc.radius * arc.end_angle.sin(),
                            ];
                            ("arc", start, end)
                        }
                        FitData::Bezier(bezier) => ("bezier", bezier.p0, bezier.p3),
                    };

                    PrimitiveCandidate {
                        primitive_type: primitive_type.to_string(),
                        start,
                        end,
                        rms_error: fit.rms_error,
                        confidence: (1.0 / (1.0 + fit.rms_error)).clamp(0.0, 1.0),
                    }
                })
            })
            .collect()
    }

    fn text_candidates(&self, image: &GrayImage) -> Vec<TextCandidate> {
        let ocr = HeuristicOcrBackend::new();
        ocr.recognize(image)
            .into_iter()
            .map(|text| {
                let accepted = text.confidence >= self.min_text_confidence;
                TextCandidate {
                    content: text.text,
                    confidence: text.confidence,
                    bbox: [
                        text.bbox.x_min as f64,
                        text.bbox.y_min as f64,
                        text.bbox.x_max as f64,
                        text.bbox.y_max as f64,
                    ],
                    rotation: text.orientation,
                    accepted,
                }
            })
            .collect()
    }

    fn symbol_candidates(&self, image: &GrayImage) -> Vec<SymbolCandidate> {
        let classifier = SymbolClassifier::new();
        classifier
            .classify(image)
            .into_iter()
            .map(|symbol| SymbolCandidate {
                symbol_type: symbol.symbol_type.to_string(),
                confidence: symbol.confidence,
                bbox: [
                    symbol.x as f64,
                    symbol.y as f64,
                    (symbol.x + symbol.width) as f64,
                    (symbol.y + symbol.height) as f64,
                ],
                rotation: symbol.rotation,
            })
            .collect()
    }

    #[cfg(test)]
    fn dimension_candidates(&self, text_candidates: &[TextCandidate]) -> Vec<DimensionCandidate> {
        let input = schema::RasterVlmInput {
            schema_version: "raster-vlm-1.0".to_string(),
            image: schema::RasterImageMetadata {
                width: 0,
                height: 0,
                color: "luma8".to_string(),
            },
            thumbnail_png_base64: None,
            polylines: Vec::new(),
            primitive_graph: Default::default(),
            text_candidates: text_candidates.to_vec(),
            symbol_candidates: Vec::new(),
        };
        HeuristicRasterVlmBackend
            .analyze(&input)
            .map(|output| output.dimension_candidates)
            .unwrap_or_default()
    }
}

fn fuse_dimensions(mut candidates: Vec<DimensionCandidate>) -> Vec<DimensionCandidate> {
    candidates.sort_by(|a, b| {
        source_priority(&b.source)
            .cmp(&source_priority(&a.source))
            .then_with(|| b.confidence.total_cmp(&a.confidence))
    });
    candidates
}

fn fuse_semantics(mut candidates: Vec<SemanticCandidate>) -> Vec<SemanticCandidate> {
    candidates.sort_by(|a, b| {
        a.target_id
            .cmp(&b.target_id)
            .then_with(|| source_priority(&b.source).cmp(&source_priority(&a.source)))
            .then_with(|| b.confidence.total_cmp(&a.confidence))
    });
    candidates
}

fn source_priority(source: &str) -> u8 {
    if source.contains("vlm") || source.contains("http") {
        3
    } else if source.contains("vector_graph") {
        2
    } else {
        1
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{GrayImage, Luma};

    #[test]
    fn extracts_rule_based_geometry_semantics() {
        let extractor = RasterSemanticExtractor::new();
        let image = GrayImage::new(16, 16);
        let polylines = vec![vec![[0.0, 0.0], [120.0, 0.0]]];

        let result = extractor.extract(&image, &polylines);

        assert_eq!(result.semantic_candidates.len(), 1);
        assert_eq!(result.semantic_candidates[0].semantic_type, "hard_wall");
        assert_eq!(result.semantic_candidates[0].source, "raster_semantic_rule");
    }

    #[test]
    fn parses_dimension_candidates_from_text() {
        let extractor = RasterSemanticExtractor::new();
        let text = vec![TextCandidate {
            content: "100±0.5".to_string(),
            confidence: 0.9,
            bbox: [1.0, 2.0, 30.0, 12.0],
            rotation: 0.0,
            accepted: true,
        }];

        let dimensions = extractor.dimension_candidates(&text);

        assert_eq!(dimensions.len(), 1);
        assert_eq!(dimensions[0].nominal_value, Some(100.0));
        assert_eq!(dimensions[0].upper_deviation, Some(0.5));
        assert_eq!(dimensions[0].source, "heuristic_vlm");
    }

    #[test]
    fn full_extraction_returns_structured_candidate_sets() {
        let extractor = RasterSemanticExtractor::new();
        let mut image = GrayImage::new(32, 32);
        for x in 4..28 {
            image.put_pixel(x, 16, Luma([0]));
        }
        let polylines = vec![vec![[4.0, 16.0], [28.0, 16.0]]];

        let result = extractor.extract(&image, &polylines);

        assert!(!result.primitive_candidates.is_empty());
        assert!(!result.semantic_candidates.is_empty());
    }
}
