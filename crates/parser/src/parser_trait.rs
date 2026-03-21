//! DXF 解析器 Trait 抽象
//!
//! ## 设计目标
//!
//! 1. **多态切换**：支持不同解析器实现（同步/异步/缓存）的无缝切换
//! 2. **向后兼容**：保持与现有 `DxfParser` API 的兼容性
//! 3. **可扩展性**：易于添加新的解析器实现
//!
//! ## 架构设计
//!
//! ```text
//! DxfParserTrait (Trait)
//! ├── SyncDxfParser (同步解析器 - 现有实现)
//! ├── AsyncDxfParser (异步解析器 - 新增)
//! ├── CachedDxfParser (缓存解析器 - 新增)
//! └── CompositeDxfParser (组合解析器 - 缓存 + 异步)
//! ```
//!
//! ## 使用示例
//!
//! ```rust
//! use parser::parser_trait::{DxfParserTrait, SyncDxfParser};
//!
//! // 使用同步解析器（向后兼容）
//! let parser = SyncDxfParser::default();
//! let (entities, report) = parser.parse_file_with_report("file.dxf")?;
//!
//! // 使用异步解析器（高性能）
//! #[tokio::main]
//! async fn example() {
//!     let parser = AsyncDxfParser::default();
//!     let (entities, report) = parser.parse_file_with_report("file.dxf").await?;
//! }
//! ```

use crate::{DxfConfig, DxfParseReport, DxfParser};
use common_types::{RawEntity, CadError};
use std::path::Path;

// ============================================================================
// 核心 Trait 定义
// ============================================================================

/// DXF 解析器 Trait
///
/// 所有解析器实现必须实现此 Trait
///
/// ## 注意：dyn 兼容性
///
/// 由于方法使用 `impl AsRef<Path>` 泛型参数，此 trait 不是 dyn compatible。
/// 工厂函数返回具体类型而非 `Box<dyn DxfParserTrait>`。
#[async_trait::async_trait]
pub trait DxfParserTrait: Send + Sync {
    /// 同步解析 DXF 文件
    fn parse_file(&self, path: impl AsRef<Path>) -> Result<Vec<RawEntity>, CadError> {
        // 默认实现：调用 parse_file_with_report 并丢弃报告
        let (entities, _report) = self.parse_file_with_report(path)?;
        Ok(entities)
    }

    /// 同步解析 DXF 文件并返回报告
    fn parse_file_with_report(&self, path: impl AsRef<Path>) -> Result<(Vec<RawEntity>, DxfParseReport), CadError>;

    /// 异步解析 DXF 文件
    async fn parse_file_async(&self, path: impl AsRef<Path> + Send) -> Result<Vec<RawEntity>, CadError> {
        // 默认实现：回退到同步版本
        self.parse_file(path)
    }

    /// 异步解析 DXF 文件并返回报告
    async fn parse_file_with_report_async(&self, path: impl AsRef<Path> + Send) -> Result<(Vec<RawEntity>, DxfParseReport), CadError> {
        // 默认实现：回退到同步版本
        let (entities, report) = self.parse_file_with_report(path)?;
        Ok((entities, report))
    }

    /// 解析 DXF 字节（ASCII 格式）
    fn parse_bytes(&self, bytes: &[u8]) -> Result<Vec<RawEntity>, CadError> {
        // 默认实现：回退到同步版本
        self.parse_bytes_with_report(bytes).map(|(e, _)| e)
    }

    /// 解析 DXF 字节并返回报告
    fn parse_bytes_with_report(&self, bytes: &[u8]) -> Result<(Vec<RawEntity>, DxfParseReport), CadError>;

    /// 获取解析器配置
    fn config(&self) -> &DxfConfig;

    /// 获取解析器名称（用于日志和调试）
    fn name(&self) -> &'static str;
}

// ============================================================================
// 同步解析器实现（包装现有 DxfParser）
// ============================================================================

/// 同步 DXF 解析器
///
/// 包装现有的 `DxfParser` 实现，保持向后兼容
#[derive(Clone)]
pub struct SyncDxfParser {
    inner: DxfParser,
}

impl SyncDxfParser {
    /// 创建新的同步解析器
    pub fn new() -> Self {
        Self {
            inner: DxfParser::new(),
        }
    }

    /// 使用自定义配置创建解析器
    pub fn with_config(config: DxfConfig) -> Self {
        Self {
            inner: DxfParser::new().with_config(config),
        }
    }

    /// 获取内部 DxfParser 的可变引用（用于高级配置）
    pub fn inner_mut(&mut self) -> &mut DxfParser {
        &mut self.inner
    }
}

impl Default for SyncDxfParser {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait::async_trait]
impl DxfParserTrait for SyncDxfParser {
    fn parse_file_with_report(&self, path: impl AsRef<Path>) -> Result<(Vec<RawEntity>, DxfParseReport), CadError> {
        self.inner.parse_file_with_report(path)
    }

    async fn parse_file_async(&self, path: impl AsRef<Path> + Send) -> Result<Vec<RawEntity>, CadError> {
        // 同步解析器也支持异步调用（在 tokio 环境中自动 spawn_blocking）
        let path = path.as_ref().to_path_buf();
        let parser = self.inner.clone();

        tokio::task::spawn_blocking(move || {
            parser.parse_file(path)
        })
        .await
        .unwrap_or_else(|e| Err(CadError::VectorizeFailed {
            message: format!("tokio join error: {}", e),
        }))
    }

    async fn parse_file_with_report_async(&self, path: impl AsRef<Path> + Send) -> Result<(Vec<RawEntity>, DxfParseReport), CadError> {
        let path = path.as_ref().to_path_buf();
        let parser = self.inner.clone();

        tokio::task::spawn_blocking(move || {
            parser.parse_file_with_report(path)
        })
        .await
        .unwrap_or_else(|e| Err(CadError::VectorizeFailed {
            message: format!("tokio join error: {}", e),
        }))
    }

    fn parse_bytes_with_report(&self, bytes: &[u8]) -> Result<(Vec<RawEntity>, DxfParseReport), CadError> {
        // 注意：parse_bytes 不支持报告，这里创建一个空报告
        let entities = self.inner.parse_bytes(bytes)?;
        let report = DxfParseReport::default();
        Ok((entities, report))
    }

    fn config(&self) -> &DxfConfig {
        &self.inner.config
    }

    fn name(&self) -> &'static str {
        "SyncDxfParser"
    }
}

// ============================================================================
// 辅助类型：解析器选择
// ============================================================================

/// 解析器类型枚举
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParserType {
    /// 同步解析器（默认，兼容性好）
    Sync,
    /// 异步解析器（高性能，推荐）
    Async,
    /// 缓存解析器（重复读取优化）
    Cached,
    /// 自动选择（根据环境和文件特征）
    Auto,
}

impl Default for ParserType {
    fn default() -> Self {
        Self::Auto
    }
}

impl std::fmt::Display for ParserType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParserType::Sync => write!(f, "Sync"),
            ParserType::Async => write!(f, "Async"),
            ParserType::Cached => write!(f, "Cached"),
            ParserType::Auto => write!(f, "Auto"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parser_type_display() {
        assert_eq!(ParserType::Sync.to_string(), "Sync");
        assert_eq!(ParserType::Async.to_string(), "Async");
        assert_eq!(ParserType::Cached.to_string(), "Cached");
        assert_eq!(ParserType::Auto.to_string(), "Auto");
    }

    #[test]
    fn test_sync_parser_creation() {
        let parser = SyncDxfParser::new();
        assert_eq!(parser.name(), "SyncDxfParser");
    }

    #[test]
    fn test_sync_parser_with_config() {
        let config = DxfConfig::default();
        let parser = SyncDxfParser::with_config(config);
        assert_eq!(parser.name(), "SyncDxfParser");
    }
}
