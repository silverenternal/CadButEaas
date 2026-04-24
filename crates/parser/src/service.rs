//! 解析服务主模块

use std::sync::Arc;
use std::time::Instant;

use crate::dwg_parser::DwgParser;
use crate::hatch_parser::HatchParser;
use crate::parser_factory::{DxfParserEnum, ParserFactory};
use crate::pdf_parser::{PdfContent, PdfParser};
use crate::stl_parser::StlParser;
use crate::svg_parser::SvgParser;
use common_types::request::Request;
use common_types::response::Response;
use common_types::{CadError, RawEntity, Service, ServiceHealth, ServiceMetrics, ServiceVersion};
use std::path::Path;

/// 图纸解析服务
///
/// 统一入口，自动检测文件类型并分发到对应解析器
pub struct ParserService {
    dxf_parser: DxfParserEnum,
    dwg_parser: DwgParser,
    pdf_parser: PdfParser,
    hatch_parser: HatchParser,
    svg_parser: SvgParser,
    stl_parser: StlParser,
    metrics: Arc<ServiceMetrics>,
}

impl ParserService {
    pub fn new() -> Self {
        Self {
            #[cfg(feature = "ezdxf-bridge")]
            dxf_parser: ParserFactory::create_ezdxf().expect("ezdxf parser creation"),
            #[cfg(not(feature = "ezdxf-bridge"))]
            dxf_parser: ParserFactory::create_default().expect("parser creation"),
            dwg_parser: DwgParser::new(),
            pdf_parser: PdfParser::new(),
            hatch_parser: HatchParser::new(),
            svg_parser: SvgParser::new(),
            stl_parser: StlParser::new(),
            metrics: Arc::new(ServiceMetrics::new("ParserService")),
        }
    }

    /// 使用自定义 DXF 解析器创建服务
    pub fn with_dxf_parser(dxf_parser: DxfParserEnum) -> Self {
        Self {
            dxf_parser,
            dwg_parser: DwgParser::new(),
            pdf_parser: PdfParser::new(),
            hatch_parser: HatchParser::new(),
            svg_parser: SvgParser::new(),
            stl_parser: StlParser::new(),
            metrics: Arc::new(ServiceMetrics::new("ParserService")),
        }
    }

    /// 设置图层过滤器（仅 ezdxf-bridge feature 下有效）
    #[cfg(feature = "ezdxf-bridge")]
    pub fn with_layer_filter(mut self, layers: Vec<String>) -> Self {
        self.dxf_parser = self.dxf_parser.with_layer_filter(layers);
        self
    }

    /// 设置 HATCH 解析配置
    pub fn with_hatch_ignore_solid(mut self, ignore: bool) -> Self {
        self.hatch_parser = self.hatch_parser.with_ignore_solid(ignore);
        self
    }

    /// 应用 DXF 过滤配置（直接从 config crate 的字段提取）
    ///
    /// 应用 ignore_* 设置到内部解析器
    pub fn with_dxf_filter(
        self,
        ignore_text: bool,
        ignore_dimensions: bool,
        ignore_hatch: bool,
    ) -> Self {
        let dxf_parser = self
            .dxf_parser
            .with_ignore_text(ignore_text)
            .with_ignore_dimensions(ignore_dimensions)
            .with_ignore_hatch(ignore_hatch);

        Self { dxf_parser, ..self }
    }

    /// 获取服务指标
    pub fn metrics(&self) -> &ServiceMetrics {
        &self.metrics
    }

    /// 解析文件，自动检测类型
    pub fn parse_file(&self, path: impl AsRef<Path>) -> Result<ParseResult, CadError> {
        let path = path.as_ref();
        let extension = path
            .extension()
            .and_then(|e| e.to_str())
            .map(|s| s.to_lowercase())
            .ok_or_else(|| CadError::UnsupportedFormat {
                format: "unknown".to_string(),
                supported_formats: vec![
                    "dxf".to_string(),
                    "dwg".to_string(),
                    "pdf".to_string(),
                    "svg".to_string(),
                    "stl".to_string(),
                ],
            })?;

        match extension.as_str() {
            "dxf" => {
                let mut entities = self.dxf_parser.parse_file(path)?;
                // 集成 HATCH 解析（补充 dxf crate 无法直接提取的填充图案）
                if let Ok(mut hatches) = self.hatch_parser.parse_hatch_entities(path) {
                    let hatch_count = hatches.len();
                    entities.append(&mut hatches);
                    if hatch_count > 0 {
                        tracing::info!("DXF 解析: 额外提取 {} 个 HATCH 实体", hatch_count);
                    }
                }
                Ok(ParseResult::Cad(entities))
            }
            "dwg" => {
                let entities = self.dwg_parser.parse_file(path)?;
                Ok(ParseResult::Cad(entities))
            }
            "pdf" => {
                let content = self.pdf_parser.parse_file(path)?;
                Ok(ParseResult::Pdf(content))
            }
            "svg" => {
                let entities = self.svg_parser.parse_file(path)?;
                Ok(ParseResult::Cad(entities))
            }
            "stl" => {
                let entities = self.stl_parser.parse_file(path)?;
                Ok(ParseResult::Cad(entities))
            }
            _ => Err(CadError::UnsupportedFormat {
                format: extension,
                supported_formats: vec![
                    "dxf".to_string(),
                    "dwg".to_string(),
                    "pdf".to_string(),
                    "svg".to_string(),
                    "stl".to_string(),
                ],
            }),
        }
    }

    /// 解析字节，需要指定文件类型
    pub fn parse_bytes(&self, bytes: &[u8], file_type: FileType) -> Result<ParseResult, CadError> {
        match file_type {
            FileType::Dxf => {
                let entities = self.dxf_parser.parse_bytes(bytes)?;
                // parse_bytes 无法读取原始组码，跳过 HATCH 补充解析
                Ok(ParseResult::Cad(entities))
            }
            FileType::Dwg => Err(CadError::UnsupportedFormat {
                format: "dwg".to_string(),
                supported_formats: vec![
                    "DWG 字节解析不支持，请使用 parse_file() 解析 DWG 文件".to_string()
                ],
            }),
            FileType::Pdf => {
                let content = self.pdf_parser.parse_bytes(bytes)?;
                Ok(ParseResult::Pdf(content))
            }
            FileType::Svg => {
                let entities = self.svg_parser.parse_bytes(bytes)?;
                Ok(ParseResult::Cad(entities))
            }
            FileType::Stl => {
                let entities = self.stl_parser.parse_bytes(bytes)?;
                Ok(ParseResult::Cad(entities))
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

    async fn process(
        &self,
        request: Request<Self::Payload>,
    ) -> std::result::Result<Response<Self::Data>, Self::Error> {
        let start = Instant::now();
        let result = self.parse_file(&request.payload.path);
        let latency = start.elapsed().as_secs_f64() * 1000.0;

        // 记录指标
        self.metrics.record_request(result.is_ok(), latency);

        let data = result?;
        Ok(Response::success(request.id, data, latency as u64))
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
    Svg,
    Stl,
}

impl FileType {
    /// 从文件扩展名检测类型
    pub fn from_extension(ext: &str) -> Option<Self> {
        match ext.to_lowercase().as_str() {
            "dxf" => Some(FileType::Dxf),
            "dwg" => Some(FileType::Dwg),
            "pdf" => Some(FileType::Pdf),
            "svg" => Some(FileType::Svg),
            "stl" => Some(FileType::Stl),
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
        // ParserService 现在使用 DxfParserEnum（默认为 ezdxf 类型）
        let _ = service.dxf_parser.name();
    }

    #[test]
    fn test_parser_service_with_hatch_config() {
        let service = ParserService::new().with_hatch_ignore_solid(true);
        assert!(service.hatch_parser.ignores_solid());
    }

    #[test]
    fn test_parser_service_with_dxf_filter() {
        // 验证 with_dxf_filter 链式调用可以正常编译和执行
        let service = ParserService::new().with_dxf_filter(false, false, false);
        let _ = service.dxf_parser.name();
    }
}
