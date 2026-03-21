//! 损坏文件恢复增强模块
//!
//! ## 设计目标
//!
//! 1. **结构化错误报告**：详细记录每个解析错误的位置、类型和建议
//! 2. **部分恢复策略**：即使文件部分损坏，也能提取可用数据
//! 3. **错误分类**：区分致命错误、可恢复错误、警告
//! 4. **自动修复建议**：为常见错误提供可操作的修复方案
//!
//! ## 错误分类
//!
//! | 类型 | 描述 | 处理策略 |
//! |------|------|---------|
//! | Fatal | 致命错误，无法继续解析 | 终止解析，返回错误 |
//! | Recoverable | 可恢复错误，跳过损坏部分 | 记录错误，继续解析 |
//! | Warning | 警告，数据可能不准确 | 记录警告，继续解析 |
//! | Info | 信息，用户应知晓的情况 | 记录信息，不影响解析 |
//!
//! ## 使用示例
//!
//! ```rust
//! use parser::recovery::{RecoveryManager, RecoveryStrategy};
//! use parser::DxfParser;
//!
//! let parser = DxfParser::new();
//! let mut recovery = RecoveryManager::new();
//!
//! // 配置恢复策略
//! recovery.set_strategy(RecoveryStrategy::Aggressive);
//!
//! // 解析文件（带错误恢复）
//! let result = recovery.parse_with_recovery(&parser, "damaged.dxf");
//!
//! match result {
//!     Ok((entities, report)) => {
//!         println!("解析成功：{} 个实体", entities.len());
//!         println!("警告：{} 条", report.warnings.len());
//!         println!("错误：{} 条", report.issues.len());
//!     }
//!     Err(e) => {
//!         println!("解析失败：{}", e);
//!     }
//! }
//! ```

use common_types::{RawEntity, CadError, InternalErrorReason};
use crate::{DxfParser, DxfParseReport, ParseIssue, ParseIssueSeverity};
use std::path::Path;
use std::panic::{self, AssertUnwindSafe};
use tracing;

// ============================================================================
// 错误恢复策略
// ============================================================================

/// 恢复策略配置
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RecoveryStrategy {
    /// 保守策略：遇到任何错误立即终止
    Conservative,
    /// 平衡策略（默认）：跳过损坏实体，继续解析
    #[default]
    Balanced,
    /// 激进策略：尝试修复损坏数据，最大化恢复
    Aggressive,
    /// 自定义策略
    Custom,
}

impl RecoveryStrategy {
    /// 是否应该跳过损坏实体
    pub fn should_skip_damaged_entity(self) -> bool {
        match self {
            RecoveryStrategy::Conservative => false,
            RecoveryStrategy::Balanced | RecoveryStrategy::Aggressive | RecoveryStrategy::Custom => true,
        }
    }

    /// 是否应该尝试修复损坏数据
    pub fn should_attempt_repair(self) -> bool {
        match self {
            RecoveryStrategy::Aggressive | RecoveryStrategy::Custom => true,
            _ => false,
        }
    }

    /// 是否应该继续解析（遇到非致命错误时）
    pub fn should_continue_on_error(self) -> bool {
        match self {
            RecoveryStrategy::Conservative => false,
            RecoveryStrategy::Balanced | RecoveryStrategy::Aggressive | RecoveryStrategy::Custom => true,
        }
    }
}

// ============================================================================
// 损坏实体修复器
// ============================================================================

/// 实体修复器 Trait
pub trait EntityRepairer {
    /// 尝试修复损坏的实体
    fn repair_entity(&self, entity_type: &str, error_msg: &str) -> Option<RawEntity>;
    
    /// 检查实体是否有效
    fn is_valid_entity(&self, entity: &RawEntity) -> bool;
}

/// 默认实体修复器实现
#[derive(Debug, Clone, Default)]
pub struct DefaultEntityRepairer;

impl EntityRepairer for DefaultEntityRepairer {
    fn repair_entity(&self, entity_type: &str, error_msg: &str) -> Option<RawEntity> {
        // 激进策略：尝试从错误中恢复部分数据
        // 注意：这里只是示例实现，实际修复逻辑需要根据具体错误类型定制
        
        tracing::debug!("尝试修复损坏实体：type={}, error={}", entity_type, error_msg);
        
        // 当前不支持自动修复，返回 None
        // 未来可以扩展为：
        // - 修复零长度线段：使用端点平均值创建点实体
        // - 修复无效坐标：替换为默认值或相邻点
        // - 修复缺失数据：使用默认值填充
        None
    }
    
    fn is_valid_entity(&self, entity: &RawEntity) -> bool {
        // 基础验证：检查是否有 NaN 或无穷大值
        match entity {
            RawEntity::Line { start, end, .. } => {
                is_valid_point(start) && is_valid_point(end) && {
                    let len = distance_2d(start, end);
                    len > 1e-6 && len < 1e6  // 长度在合理范围内
                }
            }
            RawEntity::Polyline { points, .. } => {
                points.iter().all(|p| is_valid_point(p)) && points.len() >= 2
            }
            RawEntity::Arc { center, radius, .. } => {
                is_valid_point(center) && *radius > 0.0 && *radius < 1e6
            }
            RawEntity::Circle { center, radius, .. } => {
                is_valid_point(center) && *radius > 0.0 && *radius < 1e6
            }
            RawEntity::Text { position, height, .. } => {
                is_valid_point(position) && *height > 0.0 && *height < 1e3
            }
            RawEntity::BlockReference { .. } => true,  // 块引用不需要几何验证
            RawEntity::Dimension { .. } => true,  // 标注不需要严格验证
            RawEntity::Path { .. } => true,  // 路径不需要严格验证
            RawEntity::Hatch { boundary_paths, .. } => {
                // 验证 HATCH 边界路径
                boundary_paths.iter().all(|path| {
                    match path {
                        common_types::HatchBoundaryPath::Polyline { points, .. } => {
                            points.iter().all(|p| is_valid_point(p)) && points.len() >= 2
                        }
                        common_types::HatchBoundaryPath::Arc { center, radius, .. } => {
                            is_valid_point(center) && *radius > 0.0 && *radius < 1e6
                        }
                        common_types::HatchBoundaryPath::EllipseArc { center, major_axis, minor_axis_ratio, .. } => {
                            is_valid_point(center) && is_valid_point(major_axis) && 
                            *minor_axis_ratio > 0.0 && *minor_axis_ratio <= 1.0
                        }
                        common_types::HatchBoundaryPath::Spline { control_points, .. } => {
                            control_points.iter().all(|p| is_valid_point(p)) && control_points.len() >= 2
                        }
                    }
                })
            }
            RawEntity::XRef { .. } => {
                // P1-1: XREF 外部参照支持 - 待完整实现
                // XREF 不需要严格几何验证，仅验证基本结构
                true
            }
        }
    }
}

/// 检查点是否有效（非 NaN、非无穷大）
fn is_valid_point(point: &[f64; 2]) -> bool {
    point[0].is_finite() && point[1].is_finite()
}

/// 计算两点间距离
fn distance_2d(p1: &[f64; 2], p2: &[f64; 2]) -> f64 {
    ((p2[0] - p1[0]).powi(2) + (p2[1] - p1[1]).powi(2)).sqrt()
}

// ============================================================================
// 恢复管理器
// ============================================================================

/// 恢复管理器 - 核心协调器
pub struct RecoveryManager<R: EntityRepairer = DefaultEntityRepairer> {
    /// 恢复策略
    strategy: RecoveryStrategy,
    /// 实体修复器
    repairer: R,
    /// 最大允许错误数（超过则终止解析）
    max_errors: usize,
    /// 是否详细记录错误（用于调试）
    verbose_logging: bool,
}

impl Default for RecoveryManager {
    fn default() -> Self {
        Self::new()
    }
}

impl RecoveryManager {
    /// 创建新的恢复管理器（使用默认修复器）
    pub fn new() -> Self {
        Self {
            strategy: RecoveryStrategy::Balanced,
            repairer: DefaultEntityRepairer,
            max_errors: 100,  // 默认最多允许 100 个错误
            verbose_logging: false,
        }
    }
}

impl<R: EntityRepairer> RecoveryManager<R> {
    /// 创建新的恢复管理器（自定义修复器）
    pub fn with_repairer(repairer: R) -> Self {
        Self {
            strategy: RecoveryStrategy::Balanced,
            repairer,
            max_errors: 100,
            verbose_logging: false,
        }
    }

    /// 设置恢复策略
    pub fn with_strategy(mut self, strategy: RecoveryStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// 设置最大允许错误数
    pub fn with_max_errors(mut self, max_errors: usize) -> Self {
        self.max_errors = max_errors;
        self
    }

    /// 启用详细日志
    pub fn with_verbose_logging(mut self, verbose: bool) -> Self {
        self.verbose_logging = verbose;
        self
    }

    /// 获取当前策略
    pub fn strategy(&self) -> RecoveryStrategy {
        self.strategy
    }

    /// 设置策略
    pub fn set_strategy(&mut self, strategy: RecoveryStrategy) {
        self.strategy = strategy;
    }

    /// 解析文件（带错误恢复）
    ///
    /// # 返回
    /// - `Ok((entities, report))` - 解析成功（可能包含部分恢复的数据）
    /// - `Err(CadError)` - 解析失败（致命错误）
    pub fn parse_with_recovery<P: AsRef<Path>>(
        &self,
        parser: &DxfParser,
        path: P,
    ) -> Result<(Vec<RawEntity>, DxfParseReport), CadError> {
        let path = path.as_ref();
        
        // 使用 catch_unwind 捕获整个解析过程的 panic
        let result = panic::catch_unwind(AssertUnwindSafe(|| {
            parser.parse_file_with_report(path)
        }));

        match result {
            Ok(Ok((entities, report))) => {
                // 解析成功，验证并修复实体
                self.validate_and_fix_entities(entities, report)
            }
            Ok(Err(e)) => {
                // 解析失败，尝试部分恢复
                self.handle_parse_error(e, path)
            }
            Err(panic_info) => {
                // 捕获到 panic，记录并返回错误
                let panic_msg = panic_info
                    .downcast_ref::<String>()
                    .map(|s| s.as_str())
                    .or_else(|| panic_info.downcast_ref::<&str>().copied())
                    .unwrap_or("未知 panic");

                tracing::error!("解析过程中捕获 panic: {}", panic_msg);
                
                Err(CadError::internal(
                    InternalErrorReason::Panic {
                        message: format!("解析过程中发生未处理错误：{}", panic_msg),
                    }
                ))
            }
        }
    }

    /// 验证并修复实体
    fn validate_and_fix_entities(
        &self,
        mut entities: Vec<RawEntity>,
        mut report: DxfParseReport,
    ) -> Result<(Vec<RawEntity>, DxfParseReport), CadError> {
        let mut removed_count = 0;

        // 过滤无效实体
        entities.retain(|entity| {
            if self.repairer.is_valid_entity(entity) {
                true
            } else {
                removed_count += 1;
                
                // 记录问题
                let issue = ParseIssue::new(
                    "INVALID_ENTITY",
                    "检测到无效实体，已移除",
                    ParseIssueSeverity::Warning,
                )
                .with_entity_type(entity.entity_type_name())
                .with_layer(entity.layer().unwrap_or("unknown").to_string())
                .with_suggestion("检查源 DXF 文件是否损坏".to_string());
                
                report.issues.push(issue);
                
                false
            }
        });

        // 记录统计
        report.parse_stats.valid_entities = entities.len();
        report.parse_stats.recovered_entities = 0;  // 暂时不支持自动修复
        report.parse_stats.corrupted_entities = removed_count;
        report.parse_stats.calculate_recovery_rate();

        if self.verbose_logging {
            tracing::info!(
                "实体验证完成：保留 {} 个，移除 {} 个",
                entities.len(), removed_count
            );
        }

        Ok((entities, report))
    }

    /// 处理解析错误（尝试部分恢复）
    fn handle_parse_error(
        &self,
        error: CadError,
        path: &Path,
    ) -> Result<(Vec<RawEntity>, DxfParseReport), CadError> {
        tracing::error!("解析失败：{:?}, 路径：{:?}", error, path);

        // 根据策略决定是否继续
        if !self.strategy.should_continue_on_error() {
            return Err(error);
        }

        // 尝试创建空报告并返回
        // 未来可以扩展为：
        // - 尝试使用备用解析器（如 ODA 库）
        // - 尝试逐段解析文件
        // - 尝试从备份文件恢复
        
        let mut report = DxfParseReport::default();
        report.issues.push(ParseIssue::new(
            "PARSE_FAILED",
            format!("文件解析失败：{}", error),
            ParseIssueSeverity::Error,
        )
        .with_suggestion("尝试使用 AutoCAD 重新保存文件".to_string()));

        Err(error)
    }

    /// 解析字节（带错误恢复）
    pub fn parse_bytes_with_recovery(
        &self,
        parser: &DxfParser,
        bytes: &[u8],
    ) -> Result<(Vec<RawEntity>, DxfParseReport), CadError> {
        let result = panic::catch_unwind(AssertUnwindSafe(|| {
            parser.parse_bytes(bytes)
        }));

        match result {
            Ok(Ok(entities)) => {
                // 创建一个基本报告
                let mut report = DxfParseReport::default();
                report.entity_type_distribution.insert(
                    "parsed_from_bytes".to_string(),
                    entities.len(),
                );
                self.validate_and_fix_entities(entities, report)
            }
            Ok(Err(e)) => {
                self.handle_parse_error(e, Path::new("<bytes>"))
            }
            Err(_) => {
                Err(CadError::internal(
                    InternalErrorReason::Panic {
                        message: "解析字节时发生未处理错误".to_string(),
                    }
                ))
            }
        }
    }
}

// ============================================================================
// 辅助函数：清理 MText 内容
// ============================================================================

/// 清理 MText 中的 DXF 格式代码
///
/// DXF MText 可能包含以下格式代码：
/// - \X  换行（无段落间距）
/// - \P  新段落
/// - \n  新行
/// - \~  非换行空格
/// - \\  反斜杠本身
/// - {\  左大括号
/// - }   右大括号
/// - \Cn  颜色（n=1-255）
/// - \Ln  行间距
/// 等等
pub fn clean_mtext_content(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();

    while let Some(c) = chars.next() {
        if c == '\\' {
            // 转义序列
            match chars.peek() {
                Some('X') | Some('x') => {
                    chars.next();
                    result.push('\n');  // 换行
                }
                Some('P') | Some('p') => {
                    chars.next();
                    result.push('\n');  // 新段落
                    result.push('\n');
                }
                Some('n') => {
                    chars.next();
                    result.push('\n');  // 新行
                }
                Some('~') => {
                    chars.next();
                    result.push(' ');   // 非换行空格
                }
                Some('\\') => {
                    chars.next();
                    result.push('\\');  // 反斜杠本身
                }
                Some('{') => {
                    chars.next();
                    result.push('{');   // 左大括号
                }
                Some('}') => {
                    chars.next();
                    result.push('}');   // 右大括号
                }
                Some('C') | Some('c') => {
                    // 颜色代码 \Cn
                    chars.next();
                    // 跳过数字
                    while let Some(c) = chars.peek() {
                        if c.is_ascii_digit() {
                            chars.next();
                        } else {
                            break;
                        }
                    }
                }
                Some('L') | Some('l') => {
                    // 行间距代码 \Ln
                    chars.next();
                    // 跳过数字
                    while let Some(c) = chars.peek() {
                        if c.is_ascii_digit() {
                            chars.next();
                        } else {
                            break;
                        }
                    }
                }
                _ => {
                    // 未知转义，保留反斜杠
                    result.push(c);
                }
            }
        } else if c == '{' || c == '}' {
            // 跳过未转义的大括号（格式控制符）
            continue;
        } else {
            result.push(c);
        }
    }

    result.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_recovery_strategy() {
        assert!(!RecoveryStrategy::Conservative.should_skip_damaged_entity());
        assert!(RecoveryStrategy::Balanced.should_skip_damaged_entity());
        assert!(RecoveryStrategy::Aggressive.should_skip_damaged_entity());

        assert!(!RecoveryStrategy::Conservative.should_attempt_repair());
        assert!(!RecoveryStrategy::Balanced.should_attempt_repair());
        assert!(RecoveryStrategy::Aggressive.should_attempt_repair());

        assert!(!RecoveryStrategy::Conservative.should_continue_on_error());
        assert!(RecoveryStrategy::Balanced.should_continue_on_error());
        assert!(RecoveryStrategy::Aggressive.should_continue_on_error());
    }

    #[test]
    fn test_entity_validation() {
        let repairer = DefaultEntityRepairer;

        // 有效线段
        let valid_line = RawEntity::Line {
            start: [0.0, 0.0],
            end: [10.0, 10.0],
            metadata: common_types::EntityMetadata::new(),
            semantic: Some(common_types::BoundarySemantic::HardWall),
        };
        assert!(repairer.is_valid_entity(&valid_line));

        // 无效线段（零长度）
        let invalid_line = RawEntity::Line {
            start: [5.0, 5.0],
            end: [5.0, 5.0],
            metadata: common_types::EntityMetadata::new(),
            semantic: Some(common_types::BoundarySemantic::HardWall),
        };
        assert!(!repairer.is_valid_entity(&invalid_line));

        // 无效线段（NaN 坐标）
        let nan_line = RawEntity::Line {
            start: [f64::NAN, 0.0],
            end: [10.0, 10.0],
            metadata: common_types::EntityMetadata::new(),
            semantic: Some(common_types::BoundarySemantic::HardWall),
        };
        assert!(!repairer.is_valid_entity(&nan_line));

        // 无效圆（负半径）
        let invalid_circle = RawEntity::Circle {
            center: [0.0, 0.0],
            radius: -5.0,
            metadata: common_types::EntityMetadata::new(),
            semantic: Some(common_types::BoundarySemantic::HardWall),
        };
        assert!(!repairer.is_valid_entity(&invalid_circle));
    }

    #[test]
    fn test_mtext_cleaning() {
        assert_eq!(clean_mtext_content("Hello\\PWorld"), "Hello\n\nWorld");
        assert_eq!(clean_mtext_content("Line1\\XLine2"), "Line1\nLine2");
        assert_eq!(clean_mtext_content("Test\\~Space"), "Test Space");
        assert_eq!(clean_mtext_content("Path\\\\File"), "Path\\File");
        assert_eq!(clean_mtext_content("Color\\C1Text"), "ColorText");
    }

    #[test]
    fn test_recovery_manager_creation() {
        let manager = RecoveryManager::new();
        assert_eq!(manager.strategy(), RecoveryStrategy::Balanced);

        let manager = RecoveryManager::default();
        assert_eq!(manager.strategy(), RecoveryStrategy::Balanced);

        let manager = RecoveryManager::new()
            .with_strategy(RecoveryStrategy::Aggressive);
        assert_eq!(manager.strategy(), RecoveryStrategy::Aggressive);
    }
}
