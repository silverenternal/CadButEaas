//! 解析服务主模块

use std::sync::Arc;
use std::time::Instant;

use common_types::{RawEntity, CadError, Service, ServiceHealth, ServiceVersion, ServiceMetrics};
use common_types::request::Request;
use common_types::response::Response;
use crate::dxf_parser::DxfParser;
use crate::pdf_parser::{PdfParser, PdfContent};
use std::path::Path;

/// 图纸解析服务
///
/// 统一入口，自动检测文件类型并分发到对应解析器
pub struct ParserService {
    dxf_parser: DxfParser,
    pdf_parser: PdfParser,
    metrics: Arc<ServiceMetrics>,
}

impl ParserService {
    pub fn new() -> Self {
        Self {
            dxf_parser: DxfParser::new(),
            pdf_parser: PdfParser::new(),
            metrics: Arc::new(ServiceMetrics::new("ParserService")),
        }
    }

    /// 设置图层过滤器
    pub fn with_layer_filter(mut self, layers: Vec<String>) -> Self {
        self.dxf_parser = self.dxf_parser.with_layer_filter(layers.clone());
        self
    }

    /// 获取服务指标
    pub fn metrics(&self) -> &ServiceMetrics {
        &self.metrics
    }

    /// 解析文件，自动检测类型
    pub fn parse_file(&self, path: impl AsRef<Path>) -> Result<ParseResult, CadError> {
        let path = path.as_ref();
        let extension = path.extension()
            .and_then(|e| e.to_str())
            .map(|s| s.to_lowercase())
            .ok_or_else(|| CadError::UnsupportedFormat {
                format: "unknown".to_string(),
                supported_formats: vec!["dxf".to_string(), "pdf".to_string()],
            })?;

        match extension.as_str() {
            "dxf" => {
                let entities = self.dxf_parser.parse_file(path)?;
                Ok(ParseResult::Cad(entities))
            }
            "dwg" => {
                // DWG 是专有格式，需要 ODA 或 LibreDWG 支持
                Err(CadError::UnsupportedFormat {
                    format: "dwg".to_string(),
                    supported_formats: vec!["dxf".to_string(), "pdf".to_string()],
                })
            }
            "pdf" => {
                let content = self.pdf_parser.parse_file(path)?;
                Ok(ParseResult::Pdf(content))
            }
            _ => Err(CadError::UnsupportedFormat {
                format: extension,
                supported_formats: vec!["dxf".to_string(), "pdf".to_string()],
            }),
        }
    }

    /// 解析字节，需要指定文件类型
    pub fn parse_bytes(&self, bytes: &[u8], file_type: FileType) -> Result<ParseResult, CadError> {
        match file_type {
            FileType::Dxf => {
                let entities = self.dxf_parser.parse_bytes(bytes)?;
                Ok(ParseResult::Cad(entities))
            }
            FileType::Dwg => {
                Err(CadError::UnsupportedFormat {
                    format: "dwg".to_string(),
                    supported_formats: vec!["dxf".to_string(), "pdf".to_string()],
                })
            }
            FileType::Pdf => {
                let content = self.pdf_parser.parse_bytes(bytes)?;
                Ok(ParseResult::Pdf(content))
            }
        }
    }
}

impl Default for ParserService {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Service Trait 实现
// ============================================================================

#[async_trait::async_trait]
impl Service for ParserService {
    type Payload = ParseRequest;
    type Data = ParseResult;
    type Error = CadError;

    async fn process(&self, request: Request<Self::Payload>) -> std::result::Result<Response<Self::Data>, Self::Error> {
        let start = Instant::now();
        let result = self.parse_file(&request.payload.path);
        let latency = start.elapsed().as_secs_f64() * 1000.0;

        // 记录指标
        self.metrics.record_request(result.is_ok(), latency);

        let data = result?;
        Ok(Response::success(
            request.id,
            data,
            latency as u64,
        ))
    }

    fn health_check(&self) -> ServiceHealth {
        ServiceHealth::healthy(self.version().semver.clone())
    }

    fn version(&self) -> ServiceVersion {
        ServiceVersion::new(env!("CARGO_PKG_VERSION"))
    }

    fn service_name(&self) -> &'static str {
        "ParserService"
    }

    fn metrics(&self) -> &ServiceMetrics {
        &self.metrics
    }
}

/// 解析服务请求
#[derive(Debug, Clone)]
pub struct ParseRequest {
    pub path: String,
}

impl ParseRequest {
    pub fn new(path: impl Into<String>) -> Self {
        Self { path: path.into() }
    }
}

/// 文件类型枚举
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileType {
    Dxf,
    Dwg,
    Pdf,
}

impl FileType {
    /// 从文件扩展名检测类型
    pub fn from_extension(ext: &str) -> Option<Self> {
        match ext.to_lowercase().as_str() {
            "dxf" => Some(FileType::Dxf),
            "dwg" => Some(FileType::Dwg),
            "pdf" => Some(FileType::Pdf),
            _ => None,
        }
    }
}

/// 解析结果
#[derive(Debug)]
pub enum ParseResult {
    /// CAD 矢量数据
    Cad(Vec<RawEntity>),
    /// PDF 内容 (可能包含矢量 + 光栅)
    Pdf(PdfContent),
}

impl ParseResult {
    /// 获取矢量实体
    pub fn into_entities(self) -> Vec<RawEntity> {
        match self {
            ParseResult::Cad(entities) => entities,
            ParseResult::Pdf(content) => content.vector_entities,
        }
    }

    /// 借用矢量实体引用
    pub fn as_entities(&self) -> Vec<&RawEntity> {
        match self {
            ParseResult::Cad(entities) => entities.iter().collect(),
            ParseResult::Pdf(content) => content.vector_entities.iter().collect(),
        }
    }

    /// 是否为光栅内容
    pub fn has_raster(&self) -> bool {
        match self {
            ParseResult::Cad(_) => false,
            ParseResult::Pdf(content) => !content.raster_images.is_empty(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parser_service_new() {
        let service = ParserService::new();
        assert!(service.dxf_parser.layer_filter.is_none());
    }
}
