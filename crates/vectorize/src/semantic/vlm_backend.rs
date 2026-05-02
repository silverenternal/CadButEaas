//! Pluggable raster VLM backend implementations.

use std::io::{Read, Write};
use std::net::{Shutdown, TcpStream};
use std::time::{Duration, Instant};

use base64::{engine::general_purpose, Engine as _};
use common_types::Polyline;
use image::{codecs::png::PngEncoder, ColorType, GrayImage, ImageEncoder};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::algorithms::{BoundingBox, DimensionAnalyzer, TextRecognition};
use crate::pipeline::{DimensionCandidate, SemanticCandidate, SymbolCandidate, TextCandidate};

use super::schema::{
    RasterImageMetadata, RasterPrimitiveEdge, RasterPrimitiveGraph, RasterPrimitiveNode,
    RasterVlmInput, RasterVlmOutput,
};

#[derive(Debug, Error)]
pub enum RasterVlmError {
    #[error("HTTP endpoint must start with http://")]
    UnsupportedEndpoint,
    #[error("invalid HTTP endpoint: {0}")]
    InvalidEndpoint(String),
    #[error("HTTP transport failed: {0}")]
    Transport(String),
    #[error("HTTP backend returned status {status}: {body}")]
    HttpStatus { status: u16, body: String },
    #[error("VLM JSON parse failed: {0}")]
    Json(String),
    #[error("image encoding failed: {0}")]
    Image(String),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RasterVlmBackendKind {
    Disabled,
    Heuristic,
    Http,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RasterVlmBackendConfig {
    pub kind: RasterVlmBackendKind,
    pub endpoint: Option<String>,
    pub timeout_ms: u64,
    pub max_thumbnail_px: u32,
}

impl Default for RasterVlmBackendConfig {
    fn default() -> Self {
        Self {
            kind: RasterVlmBackendKind::Heuristic,
            endpoint: Some("http://127.0.0.1:8765/analyze_raster".to_string()),
            timeout_ms: 5_000,
            max_thumbnail_px: 1024,
        }
    }
}

pub trait RasterVlmBackend: Send + Sync {
    fn analyze(&self, input: &RasterVlmInput) -> Result<RasterVlmOutput, RasterVlmError>;
}

#[derive(Debug, Clone, Default)]
pub struct DisabledRasterVlmBackend;

impl RasterVlmBackend for DisabledRasterVlmBackend {
    fn analyze(&self, _input: &RasterVlmInput) -> Result<RasterVlmOutput, RasterVlmError> {
        Ok(RasterVlmOutput::empty("disabled"))
    }
}

#[derive(Debug, Clone, Default)]
pub struct HeuristicRasterVlmBackend;

impl RasterVlmBackend for HeuristicRasterVlmBackend {
    fn analyze(&self, input: &RasterVlmInput) -> Result<RasterVlmOutput, RasterVlmError> {
        let mut output = RasterVlmOutput::empty("heuristic");
        output.dimension_candidates = heuristic_dimensions(&input.text_candidates);
        output.semantic_candidates = heuristic_semantics(&input.polylines);
        Ok(output)
    }
}

#[derive(Debug, Clone)]
pub struct HttpRasterVlmBackend {
    endpoint: String,
    timeout: Duration,
}

impl HttpRasterVlmBackend {
    pub fn new(endpoint: impl Into<String>, timeout: Duration) -> Self {
        Self {
            endpoint: endpoint.into(),
            timeout,
        }
    }
}

impl RasterVlmBackend for HttpRasterVlmBackend {
    fn analyze(&self, input: &RasterVlmInput) -> Result<RasterVlmOutput, RasterVlmError> {
        let request =
            serde_json::to_vec(input).map_err(|err| RasterVlmError::Json(err.to_string()))?;
        let (host, port, path) = parse_http_endpoint(&self.endpoint)?;
        let addr = format!("{host}:{port}");
        let start = Instant::now();
        let mut stream = TcpStream::connect(&addr)
            .map_err(|err| RasterVlmError::Transport(format!("connect {addr}: {err}")))?;
        stream
            .set_read_timeout(Some(self.timeout))
            .map_err(|err| RasterVlmError::Transport(err.to_string()))?;
        stream
            .set_write_timeout(Some(self.timeout))
            .map_err(|err| RasterVlmError::Transport(err.to_string()))?;

        let header = format!(
            "POST {path} HTTP/1.1\r\nHost: {host}\r\nContent-Type: application/json\r\nAccept: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            request.len()
        );
        stream
            .write_all(header.as_bytes())
            .and_then(|_| stream.write_all(&request))
            .map_err(|err| RasterVlmError::Transport(err.to_string()))?;
        stream
            .shutdown(Shutdown::Write)
            .map_err(|err| RasterVlmError::Transport(err.to_string()))?;

        let mut response = Vec::new();
        stream
            .read_to_end(&mut response)
            .map_err(|err| RasterVlmError::Transport(err.to_string()))?;
        let response = String::from_utf8_lossy(&response);
        let (status, body) = split_http_response(&response)?;
        if !(200..300).contains(&status) {
            return Err(RasterVlmError::HttpStatus {
                status,
                body: body.chars().take(512).collect(),
            });
        }
        let mut output: RasterVlmOutput =
            serde_json::from_str(body).map_err(|err| RasterVlmError::Json(err.to_string()))?;
        output
            .model_info
            .latency_ms
            .get_or_insert(start.elapsed().as_millis() as u64);
        Ok(output)
    }
}

pub fn build_vlm_input(
    image: &GrayImage,
    polylines: &[Polyline],
    text_candidates: &[TextCandidate],
    symbol_candidates: &[SymbolCandidate],
    max_thumbnail_px: u32,
) -> Result<RasterVlmInput, RasterVlmError> {
    let thumbnail_png_base64 = if max_thumbnail_px == 0 {
        None
    } else {
        Some(encode_thumbnail_png(image, max_thumbnail_px)?)
    };

    Ok(RasterVlmInput {
        schema_version: "raster-vlm-1.0".to_string(),
        image: RasterImageMetadata {
            width: image.width(),
            height: image.height(),
            color: "luma8".to_string(),
        },
        thumbnail_png_base64,
        polylines: polylines.to_vec(),
        primitive_graph: build_primitive_graph(polylines),
        text_candidates: text_candidates.to_vec(),
        symbol_candidates: symbol_candidates.to_vec(),
    })
}

fn build_primitive_graph(polylines: &[Polyline]) -> RasterPrimitiveGraph {
    let nodes = polylines
        .iter()
        .enumerate()
        .filter_map(|(id, polyline)| primitive_node(id, polyline))
        .collect::<Vec<_>>();
    let mut edges = Vec::new();
    for left in &nodes {
        for right in &nodes {
            if left.id >= right.id {
                continue;
            }
            if let Some(relation) = primitive_relation(left, right) {
                edges.push(RasterPrimitiveEdge {
                    source: left.id,
                    target: right.id,
                    relation,
                });
            }
        }
    }
    RasterPrimitiveGraph { nodes, edges }
}

fn primitive_node(id: usize, polyline: &Polyline) -> Option<RasterPrimitiveNode> {
    if polyline.len() < 2 {
        return None;
    }
    let mut x_min = f64::INFINITY;
    let mut y_min = f64::INFINITY;
    let mut x_max = f64::NEG_INFINITY;
    let mut y_max = f64::NEG_INFINITY;
    let mut x_sum = 0.0;
    let mut y_sum = 0.0;
    for point in polyline {
        x_min = x_min.min(point[0]);
        y_min = y_min.min(point[1]);
        x_max = x_max.max(point[0]);
        y_max = y_max.max(point[1]);
        x_sum += point[0];
        y_sum += point[1];
    }

    let first = polyline.first()?;
    let last = polyline.last()?;
    let dx = last[0] - first[0];
    let dy = last[1] - first[1];
    let length = polyline
        .windows(2)
        .map(|segment| {
            let dx = segment[1][0] - segment[0][0];
            let dy = segment[1][1] - segment[0][1];
            (dx * dx + dy * dy).sqrt()
        })
        .sum::<f64>();
    let angle_degrees = dy.atan2(dx).to_degrees();
    let orientation = if dx.abs() >= dy.abs() * 3.0 {
        "horizontal"
    } else if dy.abs() >= dx.abs() * 3.0 {
        "vertical"
    } else {
        "diagonal"
    };

    Some(RasterPrimitiveNode {
        id,
        primitive_type: "polyline".to_string(),
        bbox: [x_min, y_min, x_max, y_max],
        centroid: [x_sum / polyline.len() as f64, y_sum / polyline.len() as f64],
        length,
        angle_degrees,
        orientation: orientation.to_string(),
    })
}

fn primitive_relation(left: &RasterPrimitiveNode, right: &RasterPrimitiveNode) -> Option<String> {
    if bbox_touches(left.bbox, right.bbox, 4.0) {
        return Some("touches".to_string());
    }
    if left.orientation == right.orientation {
        return Some("parallel_to".to_string());
    }
    if ((left.orientation == "horizontal" && right.orientation == "vertical")
        || (left.orientation == "vertical" && right.orientation == "horizontal"))
        && bbox_touches(left.bbox, right.bbox, 2.0)
    {
        return Some("intersects".to_string());
    }
    None
}

fn bbox_touches(left: [f64; 4], right: [f64; 4], tolerance: f64) -> bool {
    !(left[2] < right[0] - tolerance
        || right[2] < left[0] - tolerance
        || left[3] < right[1] - tolerance
        || right[3] < left[1] - tolerance)
}

fn encode_thumbnail_png(
    image: &GrayImage,
    max_thumbnail_px: u32,
) -> Result<String, RasterVlmError> {
    let resized = if image.width().max(image.height()) > max_thumbnail_px {
        let scale = max_thumbnail_px as f32 / image.width().max(image.height()) as f32;
        let width = ((image.width() as f32 * scale).round() as u32).max(1);
        let height = ((image.height() as f32 * scale).round() as u32).max(1);
        image::imageops::resize(image, width, height, image::imageops::FilterType::Triangle)
    } else {
        image.clone()
    };

    let mut png = Vec::new();
    PngEncoder::new(&mut png)
        .write_image(
            resized.as_raw(),
            resized.width(),
            resized.height(),
            ColorType::L8.into(),
        )
        .map_err(|err| RasterVlmError::Image(err.to_string()))?;
    Ok(general_purpose::STANDARD.encode(png))
}

fn heuristic_dimensions(text_candidates: &[TextCandidate]) -> Vec<DimensionCandidate> {
    let analyzer = DimensionAnalyzer::default();
    text_candidates
        .iter()
        .filter_map(|candidate| {
            let dimension = analyzer
                .analyze_recognitions(&[TextRecognition {
                    text: candidate.content.clone(),
                    confidence: candidate.confidence,
                    bbox: BoundingBox {
                        x_min: candidate.bbox[0].max(0.0) as u32,
                        y_min: candidate.bbox[1].max(0.0) as u32,
                        x_max: candidate.bbox[2].max(0.0) as u32,
                        y_max: candidate.bbox[3].max(0.0) as u32,
                    },
                    orientation: candidate.rotation,
                }])
                .into_iter()
                .next()?;

            let has_signal = dimension.nominal_value.is_some()
                || dimension.tolerance_type.is_some()
                || dimension.geometric_type.is_some()
                || dimension.roughness.is_some()
                || !dimension.datums.is_empty();
            if !has_signal {
                return None;
            }

            Some(DimensionCandidate {
                raw_text: dimension.raw_text,
                nominal_value: dimension.nominal_value,
                tolerance_type: dimension.tolerance_type.map(|kind| kind.to_string()),
                upper_deviation: dimension.upper_deviation,
                lower_deviation: dimension.lower_deviation,
                geometric_type: dimension.geometric_type.map(|kind| kind.to_string()),
                datums: dimension.datums,
                roughness: dimension.roughness,
                bbox: candidate.bbox,
                confidence: dimension.confidence,
                source: "heuristic_vlm".to_string(),
            })
        })
        .collect()
}

fn heuristic_semantics(polylines: &[Polyline]) -> Vec<SemanticCandidate> {
    polylines
        .iter()
        .enumerate()
        .filter_map(|(idx, polyline)| {
            if polyline.len() < 2 {
                return None;
            }
            let length = polyline
                .windows(2)
                .map(|segment| {
                    let dx = segment[1][0] - segment[0][0];
                    let dy = segment[1][1] - segment[0][1];
                    (dx * dx + dy * dy).sqrt()
                })
                .sum::<f64>();
            let semantic_type = if length > 80.0 {
                "hard_wall"
            } else if length > 25.0 {
                "opening"
            } else {
                "detail_line"
            };
            Some(SemanticCandidate {
                target_id: idx,
                semantic_type: semantic_type.to_string(),
                confidence: if length > 80.0 { 0.72 } else { 0.55 },
                source: "raster_semantic_rule".to_string(),
            })
        })
        .collect()
}

fn parse_http_endpoint(endpoint: &str) -> Result<(String, u16, String), RasterVlmError> {
    let rest = endpoint
        .strip_prefix("http://")
        .ok_or(RasterVlmError::UnsupportedEndpoint)?;
    let (authority, path) = rest.split_once('/').unwrap_or((rest, ""));
    if authority.is_empty() {
        return Err(RasterVlmError::InvalidEndpoint(endpoint.to_string()));
    }
    let (host, port) = if let Some((host, port)) = authority.rsplit_once(':') {
        let port = port
            .parse::<u16>()
            .map_err(|_| RasterVlmError::InvalidEndpoint(endpoint.to_string()))?;
        (host.to_string(), port)
    } else {
        (authority.to_string(), 80)
    };
    Ok((host, port, format!("/{}", path)))
}

fn split_http_response(response: &str) -> Result<(u16, &str), RasterVlmError> {
    let (head, body) = response
        .split_once("\r\n\r\n")
        .ok_or_else(|| RasterVlmError::Transport("missing HTTP response body".to_string()))?;
    let status = head
        .lines()
        .next()
        .and_then(|line| line.split_whitespace().nth(1))
        .and_then(|status| status.parse::<u16>().ok())
        .ok_or_else(|| RasterVlmError::Transport("missing HTTP status".to_string()))?;
    Ok((status, body))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn http_backend_parses_mock_response() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let handle = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            stream
                .set_read_timeout(Some(Duration::from_millis(200)))
                .unwrap();
            let mut request = Vec::new();
            let mut chunk = [0_u8; 1024];
            while let Ok(n) = stream.read(&mut chunk) {
                if n == 0 {
                    break;
                }
                request.extend_from_slice(&chunk[..n]);
                if request_complete(&request) {
                    break;
                }
            }
            let body = r#"{"schema_version":"raster-vlm-1.0","model_info":{"backend":"http","model_name":"mock","latency_ms":1},"dimension_candidates":[],"symbol_candidates":[],"semantic_candidates":[{"target_id":0,"semantic_type":"hard_wall","confidence":0.8,"source":"vlm_http_mock"}],"warnings":[],"raw_output_summary":null}"#;
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body.len(),
                body
            );
            let _ = stream.write_all(response.as_bytes());
        });

        let backend = HttpRasterVlmBackend::new(
            format!("http://{addr}/analyze_raster"),
            Duration::from_secs(1),
        );
        let input = RasterVlmInput {
            schema_version: "raster-vlm-1.0".to_string(),
            image: RasterImageMetadata {
                width: 8,
                height: 8,
                color: "luma8".to_string(),
            },
            thumbnail_png_base64: None,
            polylines: vec![vec![[0.0, 0.0], [100.0, 0.0]]],
            primitive_graph: Default::default(),
            text_candidates: Vec::new(),
            symbol_candidates: Vec::new(),
        };

        let output = backend.analyze(&input).unwrap();
        handle.join().unwrap();
        assert_eq!(output.model_info.backend, "http");
        assert_eq!(output.semantic_candidates[0].source, "vlm_http_mock");
    }

    fn request_complete(request: &[u8]) -> bool {
        let Some(header_end) = request.windows(4).position(|window| window == b"\r\n\r\n") else {
            return false;
        };
        let headers = String::from_utf8_lossy(&request[..header_end]);
        let content_length = headers
            .lines()
            .find_map(|line| {
                let (name, value) = line.split_once(':')?;
                name.eq_ignore_ascii_case("content-length")
                    .then(|| value.trim().parse::<usize>().ok())
                    .flatten()
            })
            .unwrap_or(0);
        request.len() >= header_end + 4 + content_length
    }
}
