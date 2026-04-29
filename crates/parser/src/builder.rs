//! DXF 解析器构建器模式
//!
//! ## 设计目标
//!
//! 1. **流畅的 API 体验**：链式调用，代码更优雅
//! 2. **类型安全**：编译期检查配置完整性
//! 3. **灵活配置**：支持渐进式配置和预设模板
//! 4. **流式解析**：支持边解析边处理的流式 API
//!
//! ## 使用示例
//!
//! ### 基础构建器模式
//!
//! ```rust,ignore
//! use parser::builder::DxfParserBuilder;
//!
//! let parser = DxfParserBuilder::new()
//!     .with_layer_whitelist(vec!["WALL".to_string(), "DOOR".to_string()])
//!     .with_color_whitelist(vec![1, 7])  // 红色和黑色
//!     .with_arc_tolerance(0.05)
//!     .ignore_text(true)
//!     .build();
//!
//! let (entities, report) = parser.parse_file_with_report("floor_plan.dxf")?;
//! ```
//!
//! ### 预设模板
//!
//! ```rust,ignore
//! use parser::builder::DxfParserBuilder;
//!
//! // 墙体提取模板
//! let parser = DxfParserBuilder::for_wall_extraction()
//!     .build();
//!
//! // 家具提取模板
//! let parser = DxfParserBuilder::for_furniture_extraction()
//!     .build();
//!
//! // 完整解析模板（不过滤）
//! let parser = DxfParserBuilder::for_full_parsing()
//!     .build();
//! ```
//!
//! ### 流式解析
//!
//! ```rust,ignore
//! use parser::builder::DxfParserBuilder;
//!
//! let parser = DxfParserBuilder::new()
//!     .enable_streaming(true)
//!     .build();
//!
//! // 流式解析：每解析 100 个实体调用一次回调
//! parser.parse_file_streaming("floor_plan.dxf", |batch| {
//!     println!("处理批次：{} 个实体", batch.len());
//!     // 可以在此处进行增量处理
//!     Ok(())  // 继续解析
//! })?;
//! ```

use crate::{DxfConfig, DxfParseReport, DxfParser, EntityTypeFilter, LayerFilterMode};
use common_types::{CadError, RawEntity};
use std::path::Path;

// ============================================================================
// 构建器模式
// ============================================================================

/// DXF 解析器构建器
///
/// 提供流畅的链式 API 用于配置和创建 DxfParser
pub struct DxfParserBuilder {
    config: DxfConfig,
    tolerance: f64,
    layer_filter: Option<Vec<String>>,
    streaming_enabled: bool,
    streaming_batch_size: usize,
}

impl Default for DxfParserBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl DxfParserBuilder {
    /// 创建新的构建器
    pub fn new() -> Self {
        Self {
            config: DxfConfig::default(),
            tolerance: 0.1, // 默认 0.1mm 弦高误差
            layer_filter: None,
            streaming_enabled: false,
            streaming_batch_size: 100,
        }
    }

    /// 设置图层白名单
    pub fn with_layer_whitelist(mut self, layers: Vec<String>) -> Self {
        self.config.layer_whitelist = Some(layers);
        self
    }

    /// 添加单个图层到白名单
    pub fn add_layer(mut self, layer: impl Into<String>) -> Self {
        self.config
            .layer_whitelist
            .get_or_insert_with(Vec::new)
            .push(layer.into());
        self
    }

    /// 设置实体类型白名单
    pub fn with_entity_whitelist(mut self, entity_types: Vec<EntityTypeFilter>) -> Self {
        self.config.entity_whitelist = Some(entity_types);
        self
    }

    /// 添加单个实体类型到白名单
    pub fn add_entity_type(mut self, entity_type: EntityTypeFilter) -> Self {
        self.config
            .entity_whitelist
            .get_or_insert_with(Vec::new)
            .push(entity_type);
        self
    }

    /// 设置颜色白名单（ACI 颜色索引）
    pub fn with_color_whitelist(mut self, colors: Vec<i16>) -> Self {
        self.config.color_whitelist = Some(colors);
        self
    }

    /// 添加单个颜色到白名单
    pub fn add_color(mut self, color: i16) -> Self {
        self.config
            .color_whitelist
            .get_or_insert_with(Vec::new)
            .push(color);
        self
    }

    /// 设置线宽白名单
    pub fn with_lineweight_whitelist(mut self, lineweights: Vec<i16>) -> Self {
        self.config.lineweight_whitelist = Some(lineweights);
        self
    }

    /// 设置 ARC 离散化容差（mm）
    pub fn with_arc_tolerance(mut self, tolerance_mm: f64) -> Self {
        self.config.arc_tolerance_mm = tolerance_mm;
        self
    }

    /// 设置全局弦高容差（mm）
    pub fn with_tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// 设置是否忽略文本
    pub fn ignore_text(mut self, ignore: bool) -> Self {
        self.config.ignore_text = ignore;
        self
    }

    /// 设置是否忽略标注
    pub fn ignore_dimensions(mut self, ignore: bool) -> Self {
        self.config.ignore_dimensions = ignore;
        self
    }

    /// 设置是否忽略填充（HATCH）
    pub fn ignore_hatch(mut self, ignore: bool) -> Self {
        self.config.ignore_hatch = ignore;
        self
    }

    /// 设置是否检测座椅区块
    pub fn detect_seat_zones(mut self, detect: bool) -> Self {
        self.config.detect_seat_zones = detect;
        self
    }

    /// 设置是否简化 SPLINE
    pub fn simplify_splines(mut self, simplify: bool) -> Self {
        self.config.simplify_splines = simplify;
        self
    }

    /// 设置 SPLINE 最大控制点数
    pub fn max_spline_control_points(mut self, max_points: usize) -> Self {
        self.config.max_spline_control_points = max_points;
        self
    }

    /// 设置是否检测 3D 实体并警告
    pub fn detect_3d_entities(mut self, detect: bool) -> Self {
        self.config.detect_3d_entities = detect;
        self
    }

    /// 设置图层过滤模式
    pub fn with_layer_filter_mode(mut self, mode: LayerFilterMode) -> Self {
        self.config.layer_filter_mode = mode;
        self
    }

    /// 启用图层可见性控制
    pub fn enable_layer_visibility(mut self, enable: bool) -> Self {
        self.config.enable_layer_visibility = enable;
        self
    }

    /// 添加自定义图层分组
    pub fn add_layer_group(
        mut self,
        pattern: impl Into<String>,
        group_name: impl Into<String>,
    ) -> Self {
        self.config
            .custom_layer_groups
            .push((pattern.into(), group_name.into()));
        self
    }

    /// 设置图层过滤器
    pub fn with_layer_filter(mut self, layers: Vec<String>) -> Self {
        self.layer_filter = Some(layers);
        self
    }

    /// 启用流式解析
    pub fn enable_streaming(mut self, enabled: bool) -> Self {
        self.streaming_enabled = enabled;
        self
    }

    /// 设置流式解析批次大小
    pub fn with_streaming_batch_size(mut self, batch_size: usize) -> Self {
        self.streaming_batch_size = batch_size;
        self
    }

    /// 构建解析器
    pub fn build(self) -> DxfParser {
        let mut parser = DxfParser::new()
            .with_tolerance(self.tolerance)
            .with_config(self.config);

        if let Some(layers) = self.layer_filter {
            parser = parser.with_layer_filter(layers);
        }

        parser
    }

    // ========================================================================
    // 预设模板
    // ========================================================================

    /// 墙体提取模板
    ///
    /// 配置：
    /// - 只解析墙体相关图层（墙、柱、梁）
    /// - 忽略文本、标注、填充
    /// - 检测 3D 实体
    pub fn for_wall_extraction() -> Self {
        Self::new()
            .with_layer_filter_mode(LayerFilterMode::WallsOnly)
            .ignore_text(true)
            .ignore_dimensions(true)
            .ignore_hatch(true)
            .detect_3d_entities(true)
    }

    /// 门窗提取模板
    ///
    /// 配置：
    /// - 只解析门窗图层
    /// - 忽略文本、标注、填充
    pub fn for_opening_extraction() -> Self {
        Self::new()
            .with_layer_filter_mode(LayerFilterMode::OpeningsOnly)
            .ignore_text(true)
            .ignore_dimensions(true)
            .ignore_hatch(true)
    }

    /// 家具提取模板
    ///
    /// 配置：
    /// - 只解析家具图层
    /// - 检测座椅区块
    /// - 忽略文本、标注
    pub fn for_furniture_extraction() -> Self {
        Self::new()
            .with_layer_filter_mode(LayerFilterMode::Furniture)
            .detect_seat_zones(true)
            .ignore_text(true)
            .ignore_dimensions(true)
    }

    /// 完整解析模板
    ///
    /// 配置：
    /// - 不过滤任何图层或实体
    /// - 保留所有数据
    pub fn for_full_parsing() -> Self {
        Self::new()
            .with_layer_filter_mode(LayerFilterMode::All)
            .ignore_text(false)
            .ignore_dimensions(false)
            .ignore_hatch(false)
    }

    /// 快速预览模板
    ///
    /// 配置：
    /// - 只解析几何实体（线、多段线、圆弧）
    /// - 忽略文本、标注、填充
    /// - 简化 SPLINE
    pub fn for_quick_preview() -> Self {
        Self::new()
            .with_entity_whitelist(vec![
                EntityTypeFilter::Line,
                EntityTypeFilter::Polyline,
                EntityTypeFilter::LwPolyline,
                EntityTypeFilter::Arc,
                EntityTypeFilter::Circle,
            ])
            .ignore_text(true)
            .ignore_dimensions(true)
            .ignore_hatch(true)
            .simplify_splines(true)
            .max_spline_control_points(100)
    }
}

// ============================================================================
// 流式解析支持
// ============================================================================

/// 流式解析结果
#[derive(Debug, Clone)]
pub struct StreamingResult {
    /// 已解析的实体总数
    pub total_entities: usize,
    /// 已处理的批次数量
    pub batches_processed: usize,
    /// 解析报告
    pub report: DxfParseReport,
}

/// 流式解析回调函数类型
///
/// 回调函数接收一批实体，返回：
/// - `Ok(())`：继续解析
/// - `Err(CadError)`：停止解析，返回错误
pub type StreamingCallback = Box<dyn FnMut(&[RawEntity]) -> Result<(), CadError> + Send>;

impl DxfParser {
    /// 流式解析文件
    ///
    /// # 参数
    /// - `path`: DXF 文件路径
    /// - `callback`: 每批实体解析完成后的回调函数
    ///
    /// # 返回
    /// - `Ok(StreamingResult)`: 流式解析成功完成
    /// - `Err(CadError)`: 解析失败或回调返回错误
    ///
    /// # 示例
    /// ```rust,ignore
    /// let parser = DxfParser::new();
    /// let result = parser.parse_file_streaming("floor_plan.dxf", |batch| {
    ///     println!("处理批次：{} 个实体", batch.len());
    ///     // 可以在此处进行增量处理、渲染等
    ///     Ok(())  // 继续解析
    /// })?;
    ///
    /// println!("解析完成：共 {} 个实体", result.total_entities);
    /// ```
    pub fn parse_file_streaming<F>(
        &self,
        path: impl AsRef<Path>,
        mut callback: F,
    ) -> Result<StreamingResult, CadError>
    where
        F: FnMut(&[RawEntity]) -> Result<(), CadError>,
    {
        let path = path.as_ref();

        // 使用标准解析获取所有实体
        let (entities, report) = self.parse_file_with_report(path)?;

        let total_entities = entities.len();
        let batch_size = 100; // 默认批次大小

        // 分批处理实体
        let mut batches_processed = 0;
        for chunk in entities.chunks(batch_size) {
            callback(chunk)?;
            batches_processed += 1;
        }

        Ok(StreamingResult {
            total_entities,
            batches_processed,
            report,
        })
    }

    /// 带进度回调的流式解析
    ///
    /// # 参数
    /// - `path`: DXF 文件路径
    /// - `batch_size`: 每批实体数量
    /// - `callback`: 回调函数（接收批次和进度信息）
    ///
    /// # 示例
    /// ```rust,ignore
    /// let parser = DxfParser::new();
    /// let result = parser.parse_file_streaming_with_progress(
    ///     "floor_plan.dxf",
    ///     50,
    ///     |batch, progress| {
    ///         println!("进度：{:.1}%, 当前批次：{} 个实体", progress * 100.0, batch.len());
    ///         Ok(())
    ///     }
    /// )?;
    /// ```
    pub fn parse_file_streaming_with_progress<F>(
        &self,
        path: impl AsRef<Path>,
        batch_size: usize,
        mut callback: F,
    ) -> Result<StreamingResult, CadError>
    where
        F: FnMut(&[RawEntity], f64) -> Result<(), CadError>,
    {
        let path = path.as_ref();

        // 使用标准解析获取所有实体
        let (entities, report) = self.parse_file_with_report(path)?;

        let total_entities = entities.len();
        let mut batches_processed = 0;

        // 分批处理实体（带进度）
        for chunk in entities.chunks(batch_size) {
            let progress =
                (batches_processed * batch_size + chunk.len()) as f64 / total_entities as f64;
            callback(chunk, progress)?;
            batches_processed += 1;
        }

        Ok(StreamingResult {
            total_entities,
            batches_processed,
            report,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builder_basic() {
        let parser = DxfParserBuilder::new()
            .with_layer_whitelist(vec!["WALL".to_string()])
            .with_color_whitelist(vec![1, 7])
            .with_arc_tolerance(0.05)
            .ignore_text(true)
            .build();

        assert!(parser.config.layer_whitelist.is_some());
        assert_eq!(parser.config.layer_whitelist.as_ref().unwrap().len(), 1);
        assert_eq!(parser.config.color_whitelist.as_ref().unwrap(), &vec![1, 7]);
        assert_eq!(parser.config.arc_tolerance_mm, 0.05);
        assert!(parser.config.ignore_text);
    }

    #[test]
    fn test_builder_add_methods() {
        let parser = DxfParserBuilder::new()
            .add_layer("WALL")
            .add_layer("DOOR")
            .add_color(1)
            .add_color(7)
            .add_entity_type(EntityTypeFilter::Line)
            .build();

        assert_eq!(parser.config.layer_whitelist.as_ref().unwrap().len(), 2);
        assert_eq!(parser.config.color_whitelist.as_ref().unwrap().len(), 2);
        assert_eq!(parser.config.entity_whitelist.as_ref().unwrap().len(), 1);
    }

    #[test]
    fn test_builder_presets() {
        let wall_parser = DxfParserBuilder::for_wall_extraction().build();
        assert_eq!(
            wall_parser.config.layer_filter_mode,
            LayerFilterMode::WallsOnly
        );
        assert!(wall_parser.config.ignore_text);

        let furniture_parser = DxfParserBuilder::for_furniture_extraction().build();
        assert_eq!(
            furniture_parser.config.layer_filter_mode,
            LayerFilterMode::Furniture
        );
        assert!(furniture_parser.config.detect_seat_zones);

        let full_parser = DxfParserBuilder::for_full_parsing().build();
        assert_eq!(full_parser.config.layer_filter_mode, LayerFilterMode::All);
        assert!(!full_parser.config.ignore_text);

        let preview_parser = DxfParserBuilder::for_quick_preview().build();
        assert!(preview_parser.config.simplify_splines);
        assert!(preview_parser.config.ignore_text);
    }

    #[test]
    fn test_builder_chain_multiple_options() {
        let parser = DxfParserBuilder::new()
            .with_layer_whitelist(vec!["WALL".to_string(), "DOOR".to_string()])
            .with_entity_whitelist(vec![EntityTypeFilter::Line, EntityTypeFilter::Arc])
            .with_color_whitelist(vec![1, 2, 7])
            .with_lineweight_whitelist(vec![1, 2, 3])
            .with_arc_tolerance(0.08)
            .with_tolerance(0.2)
            .ignore_text(true)
            .ignore_dimensions(true)
            .ignore_hatch(false)
            .detect_seat_zones(true)
            .simplify_splines(true)
            .max_spline_control_points(200)
            .detect_3d_entities(true)
            .enable_layer_visibility(true)
            .add_layer_group("A-*", "Architectural")
            .build();

        assert_eq!(parser.config.layer_whitelist.as_ref().unwrap().len(), 2);
        assert_eq!(parser.config.entity_whitelist.as_ref().unwrap().len(), 2);
        assert_eq!(parser.config.color_whitelist.as_ref().unwrap().len(), 3);
        assert_eq!(parser.config.arc_tolerance_mm, 0.08);
        assert_eq!(parser.tolerance, 0.2);
        assert!(parser.config.ignore_text);
        assert!(parser.config.ignore_dimensions);
        assert!(!parser.config.ignore_hatch);
        assert!(parser.config.detect_seat_zones);
        assert!(parser.config.simplify_splines);
        assert_eq!(parser.config.max_spline_control_points, 200);
        assert!(parser.config.detect_3d_entities);
        assert!(parser.config.enable_layer_visibility);
        assert_eq!(parser.config.custom_layer_groups.len(), 1);
    }
}
