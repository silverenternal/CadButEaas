//! Stable request/response schema for raster VLM backends.

use common_types::Polyline;
use serde::{Deserialize, Serialize};

use crate::pipeline::{DimensionCandidate, SemanticCandidate, SymbolCandidate, TextCandidate};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RasterImageMetadata {
    pub width: u32,
    pub height: u32,
    pub color: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RasterPrimitiveGraph {
    #[serde(default)]
    pub nodes: Vec<RasterPrimitiveNode>,
    #[serde(default)]
    pub edges: Vec<RasterPrimitiveEdge>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RasterPrimitiveNode {
    pub id: usize,
    pub primitive_type: String,
    pub bbox: [f64; 4],
    pub centroid: [f64; 2],
    pub length: f64,
    pub angle_degrees: f64,
    pub orientation: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RasterPrimitiveEdge {
    pub source: usize,
    pub target: usize,
    pub relation: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RasterSceneGraph {
    #[serde(default)]
    pub nodes: Vec<RasterSceneNode>,
    #[serde(default)]
    pub edges: Vec<RasterSceneEdge>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RasterSceneNode {
    pub id: usize,
    pub semantic_type: String,
    pub primitive_id: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RasterSceneEdge {
    pub source: usize,
    pub target: usize,
    pub relation: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RasterVlmInput {
    pub schema_version: String,
    pub image: RasterImageMetadata,
    pub thumbnail_png_base64: Option<String>,
    pub polylines: Vec<Polyline>,
    #[serde(default)]
    pub primitive_graph: RasterPrimitiveGraph,
    pub text_candidates: Vec<TextCandidate>,
    pub symbol_candidates: Vec<SymbolCandidate>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RasterVlmModelInfo {
    pub backend: String,
    pub model_name: Option<String>,
    pub latency_ms: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RasterVlmOutput {
    pub schema_version: String,
    pub model_info: RasterVlmModelInfo,
    #[serde(default)]
    pub dimension_candidates: Vec<DimensionCandidate>,
    #[serde(default)]
    pub symbol_candidates: Vec<SymbolCandidate>,
    #[serde(default)]
    pub semantic_candidates: Vec<SemanticCandidate>,
    #[serde(default)]
    pub scene_graph: Option<RasterSceneGraph>,
    #[serde(default)]
    pub warnings: Vec<String>,
    #[serde(default)]
    pub raw_output_summary: Option<String>,
}

impl RasterVlmOutput {
    pub fn empty(backend: impl Into<String>) -> Self {
        Self {
            schema_version: "raster-vlm-1.0".to_string(),
            model_info: RasterVlmModelInfo {
                backend: backend.into(),
                model_name: None,
                latency_ms: None,
            },
            dimension_candidates: Vec::new(),
            symbol_candidates: Vec::new(),
            semantic_candidates: Vec::new(),
            scene_graph: None,
            warnings: Vec::new(),
            raw_output_summary: None,
        }
    }
}
