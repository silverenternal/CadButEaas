//! 图纸解析服务
//!
//! # 概述
//!
//! 负责解析 CAD/PDF 文件，输出标准化几何原语（RawEntity）。
//! 支持 DXF、PDF（矢量/光栅）格式，自动检测文件类型并分发到对应解析器。
//!
//! # 支持的格式
//!
//! ## DXF（AutoCAD）
//! - **实体类型**: LINE, LWPOLYLINE, ARC, CIRCLE, SPLINE, ELLIPSE
//! - **智能图层识别**: 自动识别墙体、门窗、家具、标注图层
//! - **曲线离散化**: 使用 NURBS 库精确离散化 SPLINE/ARC，弦高误差 < 0.1mm
//! - **单位解析**: 自动读取 `$INSUNITS` 变量并转换为毫米
//!
//! ## PDF
//! - **矢量 PDF**: 提取路径、线段、圆弧等几何实体
//! - **压缩流支持**: FlateDecode, DCTDecode, LZW, RunLength
//! - **光栅降级**: 图片型 PDF 自动转矢量化处理
//!
//! # 输入输出
//!
//! ## 输入
//! - 文件路径（`.dxf`, `.pdf`）或字节数组
//! - 可选配置：`DxfConfig`（图层/颜色/线宽过滤、容差等）
//!
//! ## 输出
//! - `ParseResult::Cad(Vec<RawEntity>)`: CAD 矢量数据
//! - `ParseResult::Pdf(PdfContent)`: PDF 内容（矢量 + 光栅）
//!
//! # RawEntity 结构
//!
//! ```
//! use common_types::geometry::Point2;
//!
//! pub struct RawEntity {
//!     pub entity_type: String,        // 实体类型
//!     pub points: Vec<Point2>,        // 几何点
//!     pub layer: Option<String>,      // 图层名
//!     pub color: Option<String>,      // 颜色
//!     pub metadata: common_types::EntityMetadata, // 元数据
//! }
//! ```
//!
//! # 使用示例
//!
//! ## 解析 DXF 文件
//!
//! ```no_run
//! use parser::{ParserService, DxfConfig};
//! use std::path::Path;
//!
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let service = ParserService::new();
//!
//! // 自定义配置：只解析墙体图层
//! let mut config = DxfConfig::default();
//! config.layer_whitelist = Some(vec!["WALL".to_string(), "墙".to_string()]);
//! config.color_whitelist = Some(vec![1, 7]); // 只保留红色和黑色
//!
//! let result = service.parse_file(Path::new("floor_plan.dxf"))?;
//!
//! match result {
//!     parser::ParseResult::Cad(entities) => {
//!         println!("解析到 {} 个实体", entities.len());
//!     }
//!     parser::ParseResult::Pdf(_) => {
//!         println!("这不是 DXF 文件");
//!     }
//! }
//! # Ok(())
//! # }
//! ```
//!
//! ## 智能图层识别
//!
//! ```rust,no_run
//! use parser::DxfParser;
//!
//! # fn example() -> Result<(), common_types::error::CadError> {
//! let parser = DxfParser::new();
//!
//! // 自动检测墙体图层
//! let wall_layers = parser.detect_wall_layers("floor_plan.dxf")?;
//! println!("墙体图层：{:?}", wall_layers);
//!
//! // 自动检测门窗图层
//! let door_window_layers = parser.detect_door_window_layers("floor_plan.dxf")?;
//! # Ok(())
//! # }
//! ```
//!
//! # 颜色过滤
//!
//! 支持 ACI 颜色索引过滤：
//!
//! | ACI | 颜色 | 常见用途 |
//! |-----|------|----------|
//! | 1 | 红 | 墙体 |
//! | 2 | 黄 | 门窗 |
//! | 3 | 绿 | 家具 |
//! | 4 | 青 | 标注 |
//! | 5 | 蓝 | 文字 |
//! | 6 | 品红 | 电气 |
//! | 7 | 黑/白 | 结构 |
//!
//! ```rust
//! use parser::{DxfParser, DxfConfig};
//!
//! let mut parser = DxfParser::new();
//! parser.config.color_whitelist = Some(vec![1, 7]); // 只保留红色和黑色（墙体）
//! ```
//!
//! # 常见错误
//!
//! | 错误代码 | 说明 | 解决方法 |
//! |----------|------|----------|
//! | `E101` | DXF 文件损坏 | 用 CAD 软件重新保存 |
//! | `E102` | PDF 加密 | 移除密码保护 |
//! | `E103` | 不支持的实体类型 | 更新解析器或简化图纸 |
//! | `W101` | 单位不明确 | 手动指定单位或使用标定功能 |

// ============================================================================
// 核心模块
// ============================================================================
pub mod dxf_parser;
pub mod pdf_parser;
pub mod service;
pub mod unit_converter; // P0-2 新增：单位转换器

// ============================================================================
// DXF 解析优化模块（P3 阶段）
// ============================================================================

/// 解析器 Trait 抽象层
///
/// 提供统一的接口定义，支持不同解析器实现的多态切换
pub mod parser_trait;

/// 异步 DXF 解析器
///
/// 使用 tokio 异步 IO 和流式解析优化大文件性能
pub mod async_parser;

/// 解析结果缓存系统
///
/// 使用 LRU 策略和文件修改检测实现智能缓存
pub mod cache;

/// 解析器工厂
///
/// 根据环境和文件特征自动选择最佳解析器实现
pub mod parser_factory;

// ============================================================================
// 损坏文件恢复增强（P0-3）
// ============================================================================

/// 损坏文件恢复模块
///
/// 提供结构化错误报告和部分恢复策略
pub mod recovery;

// ============================================================================
// API 设计改进（P0-4: 构建器模式 + 流式 API）
// ============================================================================

/// 构建器模式模块
///
/// 提供流畅的链式 API 用于配置和创建 DxfParser
pub mod builder;

// ============================================================================
// 公共导出
// ============================================================================

pub use dxf_parser::{
    evaluate_layer_filter, is_indoor_fixture, DxfConfig, DxfParseReport, DxfParser,
    EntityTypeFilter, LayerFilterMode, LayerFilterResult, ParseIssue, ParseIssueSeverity,
};
pub use pdf_parser::PdfParser;
pub use service::{FileType, ParseResult, ParserService};
pub use unit_converter::UnitConverter;

// ============================================================================
// 优化模块导出（P3 阶段）
// ============================================================================

pub use async_parser::AsyncDxfParser;
pub use cache::{CacheConfig, CacheStats, DxfCache};
pub use parser_factory::{DxfParserEnum, FactoryConfig, ParserFactory};
pub use parser_trait::{DxfParserTrait, ParserType, SyncDxfParser};

// ============================================================================
// 恢复模块导出（P0-3）
// ============================================================================

pub use recovery::{
    clean_mtext_content, DefaultEntityRepairer, EntityRepairer, RecoveryManager, RecoveryStrategy,
};

// ============================================================================
// 构建器模式导出（P0-4）
// ============================================================================

pub use builder::{DxfParserBuilder, StreamingResult};

// ============================================================================
// DXF 版本兼容性导出（P1-4）
// ============================================================================

/// DXF 版本兼容性模块
///
/// 提供完整的 DXF 版本检测、特性配置和兼容性处理
pub mod dxf_version;

pub use dxf_version::{
    DxfVersion, DxfVersionDetector, DxfVersionFeatures, DxfVersionStrategy,
    VersionCompatibilityReport, VersionToleranceConfig,
};

// ============================================================================
// ezdxf 解析器桥接（Python 主路径 + Rust fallback）
// ============================================================================

/// ezdxf 解析器桥接模块
///
/// 通过 subprocess 调用 Python ezdxf 库解析 DXF 文件，
/// 解析失败时自动降级到 Rust dxf crate 解析器。
#[cfg(feature = "ezdxf-bridge")]
pub mod ezdxf_parser;

#[cfg(feature = "ezdxf-bridge")]
pub use ezdxf_parser::EzdxfParser;

// ============================================================================
// HATCH 解析器（P0-1：建筑 CAD 核心功能）
// ============================================================================

/// HATCH 填充图案解析器
///
/// 使用 acadrust crate 解析 DXF 文件中的 HATCH 实体
pub mod hatch_parser;

pub use hatch_parser::HatchParser;

// ============================================================================
// DWG 解析器（外部转换 → DXF）
// ============================================================================

/// DWG 文件解析器
///
/// 通过外部转换工具（libredwg dwg2dxf 或 ODA File Converter）
/// 将 DWG 转为 DXF，然后委托给 DXF 解析器处理
pub mod dwg_parser;

pub use dwg_parser::{DwgConverter, DwgParser};

// ============================================================================
// SVG 解析器（Web CAD 集成）
// ============================================================================

/// SVG 文件解析器
pub mod svg_parser;

pub use svg_parser::SvgParser;

// ============================================================================
// STL 解析器（3D 制造/打印）
// ============================================================================

/// STL 文件解析器（二进制/ASCII）
pub mod stl_parser;

pub use stl_parser::StlParser;
