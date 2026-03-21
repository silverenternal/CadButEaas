//! DXF 解析器工厂
//!
//! ## 设计目标
//!
//! 1. **自动选择**：根据环境和文件特征自动选择最佳解析器
//! 2. **统一入口**：提供简单的 API 创建解析器
//! 3. **灵活配置**：支持手动指定解析器类型
//!
//! ## 选择策略
//!
//! | 条件 | 推荐解析器 | 理由 |
//! |------|-----------|------|
//! | 文件 >1MB | Async + Cache | 大文件需要异步 IO 和缓存 |
//! | 文件 <1MB | Sync | 小文件异步 overhead 不划算 |
//! | 重复打开 | Cache | 二次打开从缓存读取 |
//! | 内存受限 | Sync | 缓存占用额外内存 |
//! | 高吞吐场景 | Async | 异步可并行处理多个文件 |
//!
//! ## 使用示例
//!
//! ```rust
//! use parser::parser_factory::ParserFactory;
//!
//! // 自动选择（推荐）
//! let parser = ParserFactory::create_default()?;
//! let (entities, report) = parser.parse_file_with_report("file.dxf")?;
//!
//! // 高性能模式
//! let parser = ParserFactory::create_high_performance()?;
//! ```

use crate::parser_trait::{DxfParserTrait, ParserType, SyncDxfParser};
use crate::async_parser::AsyncDxfParser;
use crate::cache::{DxfCache, CacheConfig};
use crate::{DxfConfig, DxfParseReport};
use common_types::{CadError, RawEntity};
use std::path::Path;

/// 解析器枚举（用于工厂返回）
///
/// 由于 DxfParserTrait 不是 dyn compatible，使用枚举来包装不同类型
#[derive(Clone)]
pub enum DxfParserEnum {
    /// 同步解析器
    Sync(SyncDxfParser),
    /// 异步解析器
    Async(AsyncDxfParser),
    /// 缓存解析器（同步）
    CachedSync(DxfCache<SyncDxfParser>),
    /// 缓存解析器（异步）
    CachedAsync(DxfCache<AsyncDxfParser>),
}

impl DxfParserEnum {
    /// 同步解析文件
    pub fn parse_file(&self, path: impl AsRef<Path>) -> Result<Vec<RawEntity>, CadError> {
        match self {
            DxfParserEnum::Sync(p) => p.parse_file(path),
            DxfParserEnum::Async(p) => p.parse_file(path),
            DxfParserEnum::CachedSync(p) => p.parse_file(path),
            DxfParserEnum::CachedAsync(p) => p.parse_file(path),
        }
    }

    /// 同步解析文件并返回报告
    pub fn parse_file_with_report(&self, path: impl AsRef<Path>) -> Result<(Vec<RawEntity>, DxfParseReport), CadError> {
        match self {
            DxfParserEnum::Sync(p) => p.parse_file_with_report(path),
            DxfParserEnum::Async(p) => p.parse_file_with_report(path),
            DxfParserEnum::CachedSync(p) => p.parse_file_with_report(path),
            DxfParserEnum::CachedAsync(p) => p.parse_file_with_report(path),
        }
    }

    /// 异步解析文件
    pub async fn parse_file_async(&self, path: impl AsRef<Path> + Send) -> Result<Vec<RawEntity>, CadError> {
        match self {
            DxfParserEnum::Sync(p) => p.parse_file_async(path).await,
            DxfParserEnum::Async(p) => p.parse_file_async(path).await,
            DxfParserEnum::CachedSync(p) => p.parse_file_async(path).await,
            DxfParserEnum::CachedAsync(p) => p.parse_file_async(path).await,
        }
    }

    /// 异步解析文件并返回报告
    pub async fn parse_file_with_report_async(&self, path: impl AsRef<Path> + Send) -> Result<(Vec<RawEntity>, DxfParseReport), CadError> {
        match self {
            DxfParserEnum::Sync(p) => p.parse_file_with_report_async(path).await,
            DxfParserEnum::Async(p) => p.parse_file_with_report_async(path).await,
            DxfParserEnum::CachedSync(p) => p.parse_file_with_report_async(path).await,
            DxfParserEnum::CachedAsync(p) => p.parse_file_with_report_async(path).await,
        }
    }

    /// 获取解析器名称
    pub fn name(&self) -> &'static str {
        match self {
            DxfParserEnum::Sync(p) => p.name(),
            DxfParserEnum::Async(p) => p.name(),
            DxfParserEnum::CachedSync(p) => p.name(),
            DxfParserEnum::CachedAsync(p) => p.name(),
        }
    }
}

/// 工厂配置
#[derive(Debug, Clone)]
pub struct FactoryConfig {
    /// 解析器类型
    pub parser_type: ParserType,
    /// 是否启用缓存
    pub enable_cache: bool,
    /// DXF 解析配置
    pub dxf_config: Option<DxfConfig>,
    /// 缓存配置
    pub cache_config: Option<CacheConfig>,
    /// 缓存内存限制（MB）
    pub cache_memory_limit_mb: Option<f64>,
    /// 异步解析器缓冲区大小
    pub async_buffer_size: usize,
}

impl Default for FactoryConfig {
    fn default() -> Self {
        Self {
            parser_type: ParserType::Auto,
            enable_cache: true,
            dxf_config: None,
            cache_config: None,
            cache_memory_limit_mb: Some(200.0),
            async_buffer_size: 100,
        }
    }
}

impl FactoryConfig {
    /// 创建高性能配置（异步 + 缓存）
    pub fn high_performance() -> Self {
        Self {
            parser_type: ParserType::Async,
            enable_cache: true,
            cache_config: Some(CacheConfig::aggressive()),
            cache_memory_limit_mb: Some(500.0),
            ..Default::default()
        }
    }

    /// 创建低内存配置（同步，无缓存）
    pub fn low_memory() -> Self {
        Self {
            parser_type: ParserType::Sync,
            enable_cache: false,
            cache_memory_limit_mb: Some(50.0),
            ..Default::default()
        }
    }

    /// 创建平衡配置（自动选择，适度缓存）
    pub fn balanced() -> Self {
        Self {
            parser_type: ParserType::Auto,
            enable_cache: true,
            cache_config: Some(CacheConfig::default()),
            cache_memory_limit_mb: Some(200.0),
            ..Default::default()
        }
    }
}

/// DXF 解析器工厂
///
/// 提供统一的 API 创建不同类型的 DXF 解析器
pub struct ParserFactory;

impl ParserFactory {
    /// 创建默认解析器（自动选择）
    pub fn create_default() -> Result<DxfParserEnum, CadError> {
        Self::with_config(FactoryConfig::default())
    }

    /// 使用配置创建解析器
    pub fn with_config(config: FactoryConfig) -> Result<DxfParserEnum, CadError> {
        let parser: DxfParserEnum = match config.parser_type {
            ParserType::Auto => {
                // Auto 模式：默认使用 Async + 缓存
                if config.enable_cache {
                    let inner = config.dxf_config
                        .clone()
                        .map(AsyncDxfParser::with_config)
                        .unwrap_or_else(AsyncDxfParser::new)
                        .with_stream_buffer_size(config.async_buffer_size);
                    let cache_config = config.cache_config.unwrap_or_else(|| {
                        let mut cfg = CacheConfig::default();
                        if let Some(limit) = config.cache_memory_limit_mb {
                            cfg.max_memory_mb = Some(limit);
                        }
                        cfg
                    });
                    DxfParserEnum::CachedAsync(DxfCache::with_config(inner, cache_config))
                } else {
                    let inner = config.dxf_config
                        .clone()
                        .map(AsyncDxfParser::with_config)
                        .unwrap_or_else(AsyncDxfParser::new)
                        .with_stream_buffer_size(config.async_buffer_size);
                    DxfParserEnum::Async(inner)
                }
            }
            ParserType::Sync => {
                let inner = config.dxf_config
                    .clone()
                    .map(SyncDxfParser::with_config)
                    .unwrap_or_else(SyncDxfParser::new);
                if config.enable_cache {
                    let cache_config = config.cache_config.unwrap_or_else(|| {
                        let mut cfg = CacheConfig::default();
                        if let Some(limit) = config.cache_memory_limit_mb {
                            cfg.max_memory_mb = Some(limit);
                        }
                        cfg
                    });
                    DxfParserEnum::CachedSync(DxfCache::with_config(inner, cache_config))
                } else {
                    DxfParserEnum::Sync(inner)
                }
            }
            ParserType::Async => {
                let inner = config.dxf_config
                    .clone()
                    .map(AsyncDxfParser::with_config)
                    .unwrap_or_else(AsyncDxfParser::new)
                    .with_stream_buffer_size(config.async_buffer_size);
                if config.enable_cache {
                    let cache_config = config.cache_config.unwrap_or_else(|| {
                        let mut cfg = CacheConfig::default();
                        if let Some(limit) = config.cache_memory_limit_mb {
                            cfg.max_memory_mb = Some(limit);
                        }
                        cfg
                    });
                    DxfParserEnum::CachedAsync(DxfCache::with_config(inner, cache_config))
                } else {
                    DxfParserEnum::Async(inner)
                }
            }
            ParserType::Cached => {
                // Cached 类型：默认使用 Sync + 缓存
                let inner = config.dxf_config
                    .clone()
                    .map(SyncDxfParser::with_config)
                    .unwrap_or_else(SyncDxfParser::new);
                let cache_config = config.cache_config.unwrap_or_else(|| {
                    let mut cfg = CacheConfig::default();
                    if let Some(limit) = config.cache_memory_limit_mb {
                        cfg.max_memory_mb = Some(limit);
                    }
                    cfg
                });
                DxfParserEnum::CachedSync(DxfCache::with_config(inner, cache_config))
            }
        };

        Ok(parser)
    }

    /// 根据文件特征创建解析器
    ///
    /// 自动检测文件大小并选择最佳解析器
    pub fn create_for_file(path: impl AsRef<Path>) -> Result<DxfParserEnum, CadError> {
        Self::create_for_file_with_config(path, FactoryConfig::default())
    }

    /// 根据文件特征和配置创建解析器
    pub fn create_for_file_with_config(
        path: impl AsRef<Path>,
        mut config: FactoryConfig,
    ) -> Result<DxfParserEnum, CadError> {
        let path = path.as_ref();
        
        // 检测文件大小
        let file_size = std::fs::metadata(path)
            .map(|m| m.len())
            .unwrap_or(0);
        
        // 大文件（>1MB）使用异步 + 缓存
        if file_size > 1024 * 1024 {
            config.parser_type = ParserType::Async;
            config.enable_cache = true;
            
            // 大文件使用更大的缓冲区
            if config.async_buffer_size < 200 {
                config.async_buffer_size = 200;
            }
        }
        
        Self::with_config(config)
    }

    /// 创建同步解析器（向后兼容）
    pub fn create_sync() -> Result<DxfParserEnum, CadError> {
        let config = FactoryConfig {
            parser_type: ParserType::Sync,
            enable_cache: false,
            ..Default::default()
        };
        Self::with_config(config)
    }

    /// 创建异步解析器
    pub fn create_async() -> Result<DxfParserEnum, CadError> {
        let config = FactoryConfig {
            parser_type: ParserType::Async,
            enable_cache: false,
            ..Default::default()
        };
        Self::with_config(config)
    }

    /// 创建缓存解析器
    pub fn create_cached() -> Result<DxfParserEnum, CadError> {
        let config = FactoryConfig {
            parser_type: ParserType::Cached,
            enable_cache: true,
            ..Default::default()
        };
        Self::with_config(config)
    }

    /// 创建高性能解析器（异步 + 激进缓存）
    pub fn create_high_performance() -> Result<DxfParserEnum, CadError> {
        Self::with_config(FactoryConfig::high_performance())
    }

    /// 创建低内存解析器
    pub fn create_low_memory() -> Result<DxfParserEnum, CadError> {
        Self::with_config(FactoryConfig::low_memory())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_factory_config() {
        let config = FactoryConfig::default();
        assert_eq!(config.parser_type, ParserType::Auto);
        assert!(config.enable_cache);

        let hp = FactoryConfig::high_performance();
        assert_eq!(hp.parser_type, ParserType::Async);

        let lm = FactoryConfig::low_memory();
        assert_eq!(lm.parser_type, ParserType::Sync);
        assert!(!lm.enable_cache);
    }

    #[test]
    fn test_create_sync() {
        let parser = ParserFactory::create_sync().unwrap();
        assert_eq!(parser.name(), "SyncDxfParser");
    }

    #[test]
    fn test_create_async() {
        let parser = ParserFactory::create_async().unwrap();
        assert_eq!(parser.name(), "AsyncDxfParser");
    }

    #[test]
    fn test_create_cached() {
        let parser = ParserFactory::create_cached().unwrap();
        assert_eq!(parser.name(), "CachedDxfParser");
    }

    #[test]
    fn test_create_default() {
        let parser = ParserFactory::create_default().unwrap();
        // 默认应该是 Async 或 Cached
        let name = parser.name();
        assert!(name == "AsyncDxfParser" || name == "CachedDxfParser");
    }
}
