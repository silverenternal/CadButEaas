//! DXF 版本兼容性处理
//!
//! # 概述
//!
//! 本模块提供完整的 DXF 版本兼容性支持，覆盖从 AutoCAD R12 到 R2018 的所有主要版本。
//! 通过版本检测、特性配置和实体兼容性处理，确保不同版本 DXF 文件的正确解析。
//!
//! # 支持的 DXF 版本
//!
//! | 版本代码 | AutoCAD 版本 | 发布年份 | 格式支持 | 特性 |
//! |----------|--------------|----------|----------|------|
//! | AC1009 | AutoCAD R12 | 1992 | ASCII/Binary | 基础 2D 实体 |
//! | AC1012 | AutoCAD R13 | 1994 | Binary only | 3D 实体、块定义 |
//! | AC1014 | AutoCAD R14 | 1997 | Binary | 外部参照、动态块 |
//! | AC1015 | AutoCAD 2000 | 1999 | Binary | 多行文字、标注增强 |
//! | AC1018 | AutoCAD 2004 | 2003 | Binary | 动态块、参数化 |
//! | AC1021 | AutoCAD 2007 | 2006 | Binary | 3D 建模、NURBS |
//! | AC1024 | AutoCAD 2010 | 2009 | Binary | 网格建模、点云 |
//! | AC1027 | AutoCAD 2013 | 2012 | Binary | 关联标注、PDF 导入 |
//! | AC1032 | AutoCAD 2018 | 2017 | Binary | 共享坐标、点云增强 |
//!
//! # 版本检测流程
//!
//! ```text
//! 读取文件前 10 字节
//!   ↓
//! 检查二进制签名 (AC10xx)
//!   ↓
//! 匹配版本代码 → 配置解析策略
//!   ↓
//! 应用版本特定的实体处理规则
//! ```
//!
//! # 版本特性配置
//!
//! 不同 DXF 版本支持不同的实体类型和特性：
//!
//! ## R12 (AC1009)
//! - **支持**: LINE, CIRCLE, ARC, POLYLINE, TEXT
//! - **不支持**: SPLINE, ELLIPSE, MTEXT, XREF
//! - **注意**: POLYLINE 使用旧格式（2D 顶点）
//!
//! ## R13/R14 (AC1012/AC1014)
//! - **新增**: 3DFACE, BODY, XREF
//! - **变更**: POLYLINE 格式变更（3D 支持）
//!
//! ## AutoCAD 2000+ (AC1015+)
//! - **新增**: SPLINE, ELLIPSE, MTEXT, TABLE
//! - **增强**: 标注、引线、视口
//!
//! ## AutoCAD 2007+ (AC1021+)
//! - **新增**: NURBS 曲线、网格实体
//! - **增强**: 3D 实体、曲面建模

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fmt;

/// DXF 版本枚举
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DxfVersion {
    /// AutoCAD R12 (1992) - 最后支持 ASCII 的版本
    R12,
    /// AutoCAD R13 (1994) - 仅二进制格式
    R13,
    /// AutoCAD R14 (1997)
    R14,
    /// AutoCAD 2000
    V2000,
    /// AutoCAD 2004
    V2004,
    /// AutoCAD 2007
    V2007,
    /// AutoCAD 2010
    V2010,
    /// AutoCAD 2013
    V2013,
    /// AutoCAD 2018
    V2018,
    /// 未知版本
    Unknown,
}

impl DxfVersion {
    /// 从版本代码字符串解析
    ///
    /// # 参数
    /// - `version_code`: DXF 版本代码（如 "AC1009"）
    ///
    /// # 返回
    /// 对应的 DxfVersion 枚举值
    pub fn from_code(version_code: &str) -> Self {
        match version_code.trim() {
            "AC1009" => Self::R12,
            "AC1012" => Self::R13,
            "AC1014" => Self::R14,
            "AC1015" => Self::V2000,
            "AC1018" => Self::V2004,
            "AC1021" => Self::V2007,
            "AC1024" => Self::V2010,
            "AC1027" => Self::V2013,
            "AC1032" => Self::V2018,
            _ => Self::Unknown,
        }
    }

    /// 获取版本代码字符串
    pub fn to_code(&self) -> &'static str {
        match self {
            Self::R12 => "AC1009",
            Self::R13 => "AC1012",
            Self::R14 => "AC1014",
            Self::V2000 => "AC1015",
            Self::V2004 => "AC1018",
            Self::V2007 => "AC1021",
            Self::V2010 => "AC1024",
            Self::V2013 => "AC1027",
            Self::V2018 => "AC1032",
            Self::Unknown => "UNKNOWN",
        }
    }

    /// 获取 AutoCAD 版本字符串
    pub fn to_autocad_version(&self) -> &'static str {
        match self {
            Self::R12 => "AutoCAD R12",
            Self::R13 => "AutoCAD R13",
            Self::R14 => "AutoCAD R14",
            Self::V2000 => "AutoCAD 2000",
            Self::V2004 => "AutoCAD 2004",
            Self::V2007 => "AutoCAD 2007",
            Self::V2010 => "AutoCAD 2010",
            Self::V2013 => "AutoCAD 2013",
            Self::V2018 => "AutoCAD 2018",
            Self::Unknown => "Unknown Version",
        }
    }

    /// 获取发布年份
    pub fn release_year(&self) -> Option<u16> {
        match self {
            Self::R12 => Some(1992),
            Self::R13 => Some(1994),
            Self::R14 => Some(1997),
            Self::V2000 => Some(1999),
            Self::V2004 => Some(2003),
            Self::V2007 => Some(2006),
            Self::V2010 => Some(2009),
            Self::V2013 => Some(2012),
            Self::V2018 => Some(2017),
            Self::Unknown => None,
        }
    }

    /// 是否支持二进制格式
    ///
    /// R12 是最后一个支持 ASCII 格式的版本。
    /// R13 及以后版本仅支持二进制格式。
    pub fn supports_ascii(&self) -> bool {
        matches!(self, Self::R12)
    }

    /// 是否支持 3D 实体
    pub fn supports_3d_entities(&self) -> bool {
        matches!(
            self,
            Self::R13
                | Self::R14
                | Self::V2000
                | Self::V2004
                | Self::V2007
                | Self::V2010
                | Self::V2013
                | Self::V2018
        )
    }

    /// 是否支持 NURBS 曲线（SPLINE 实体）
    pub fn supports_nurbs(&self) -> bool {
        matches!(self, Self::V2007 | Self::V2010 | Self::V2013 | Self::V2018)
    }

    /// 是否支持动态块
    pub fn supports_dynamic_blocks(&self) -> bool {
        matches!(
            self,
            Self::V2004 | Self::V2007 | Self::V2010 | Self::V2013 | Self::V2018
        )
    }

    /// 是否支持外部参照（XREF）
    pub fn supports_xref(&self) -> bool {
        matches!(
            self,
            Self::R14
                | Self::V2000
                | Self::V2004
                | Self::V2007
                | Self::V2010
                | Self::V2013
                | Self::V2018
        )
    }

    /// 是否支持多行文字（MTEXT）
    pub fn supports_mtext(&self) -> bool {
        matches!(
            self,
            Self::V2000 | Self::V2004 | Self::V2007 | Self::V2010 | Self::V2013 | Self::V2018
        )
    }

    /// 是否支持椭圆（ELLIPSE）
    pub fn supports_ellipse(&self) -> bool {
        matches!(
            self,
            Self::V2000 | Self::V2004 | Self::V2007 | Self::V2010 | Self::V2013 | Self::V2018
        )
    }

    /// 获取版本兼容性评分（0-100）
    ///
    /// 用于评估文件与当前解析器的兼容程度
    pub fn compatibility_score(&self) -> u8 {
        match self {
            // 完全兼容
            Self::V2018 | Self::V2013 | Self::V2010 | Self::V2007 => 100,
            // 大部分兼容
            Self::V2004 | Self::V2000 => 90,
            // 部分兼容（需要降级处理）
            Self::R14 => 75,
            Self::R13 => 60,
            // 有限兼容（仅基础实体）
            Self::R12 => 50,
            Self::Unknown => 0,
        }
    }

    /// 获取推荐的处理策略
    pub fn recommended_strategy(&self) -> DxfVersionStrategy {
        match self {
            Self::R12 => DxfVersionStrategy::LegacyR12,
            Self::R13 | Self::R14 => DxfVersionStrategy::LegacyR13R14,
            Self::V2000 | Self::V2004 => DxfVersionStrategy::ModernV2000,
            Self::V2007 | Self::V2010 | Self::V2013 | Self::V2018 => {
                DxfVersionStrategy::ModernV2007Plus
            }
            Self::Unknown => DxfVersionStrategy::Fallback,
        }
    }

    /// 获取所有支持的版本列表
    pub fn all_supported_versions() -> Vec<Self> {
        vec![
            Self::R12,
            Self::R13,
            Self::R14,
            Self::V2000,
            Self::V2004,
            Self::V2007,
            Self::V2010,
            Self::V2013,
            Self::V2018,
        ]
    }
}

impl fmt::Display for DxfVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} ({})", self.to_code(), self.to_autocad_version())
    }
}

/// DXF 版本处理策略
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DxfVersionStrategy {
    /// R12 传统模式（仅基础 2D 实体）
    LegacyR12,
    /// R13/R14 过渡模式（基础 3D 支持）
    LegacyR13R14,
    /// 2000/2004 现代模式（完整 2D+ 标注）
    ModernV2000,
    /// 2007+ 高级模式（NURBS+3D）
    ModernV2007Plus,
    /// 回退模式（未知版本）
    Fallback,
}

impl DxfVersionStrategy {
    /// 获取策略描述
    pub fn description(&self) -> &'static str {
        match self {
            Self::LegacyR12 => "R12 传统模式：仅支持 LINE/CIRCLE/ARC/POLYLINE/TEXT",
            Self::LegacyR13R14 => "R13/R14 过渡模式：支持基础 3D 实体和块定义",
            Self::ModernV2000 => "2000/2004 现代模式：支持 SPLINE/ELLIPSE/MTEXT",
            Self::ModernV2007Plus => "2007+ 高级模式：支持 NURBS/3D 实体/动态块",
            Self::Fallback => "回退模式：尝试解析所有已知实体类型",
        }
    }

    /// 是否启用严格模式
    ///
    /// 严格模式下，遇到不支持的实体类型会报错。
    /// 宽松模式下，会跳过不支持的实体并记录警告。
    pub fn is_strict(&self) -> bool {
        matches!(self, Self::LegacyR12 | Self::LegacyR13R14)
    }
}

/// DXF 版本特性配置
#[derive(Debug, Clone, Default)]
pub struct DxfVersionFeatures {
    /// 支持的实体类型集合
    pub supported_entities: HashSet<String>,
    /// 不支持的实体类型集合（用于警告）
    pub unsupported_entities: HashSet<String>,
    /// 需要特殊处理的实体类型
    pub special_handling: HashMap<String, String>,
    /// 版本特定的容差配置
    pub tolerance_config: VersionToleranceConfig,
}

/// 版本特定容差配置
#[derive(Debug, Clone)]
pub struct VersionToleranceConfig {
    /// 点坐标容差（旧版本需要更大容差）
    pub point_tolerance: f64,
    /// 角度容差（度）
    pub angle_tolerance: f64,
    /// 弦高误差容差（用于曲线离散化）
    pub chord_height_tolerance: f64,
    /// 最小线段长度
    pub min_line_length: f64,
}

impl Default for VersionToleranceConfig {
    fn default() -> Self {
        Self {
            point_tolerance: 1e-6,
            angle_tolerance: 0.1,
            chord_height_tolerance: 0.1,
            min_line_length: 0.001,
        }
    }
}

impl VersionToleranceConfig {
    /// 根据版本创建容差配置
    pub fn for_version(version: DxfVersion) -> Self {
        match version {
            // R12 精度较低，使用更大容差
            DxfVersion::R12 => Self {
                point_tolerance: 1e-4,
                angle_tolerance: 0.5,
                chord_height_tolerance: 0.5,
                min_line_length: 0.01,
            },
            // R13/R14 中等精度
            DxfVersion::R13 | DxfVersion::R14 => Self {
                point_tolerance: 1e-5,
                angle_tolerance: 0.2,
                chord_height_tolerance: 0.2,
                min_line_length: 0.005,
            },
            // 现代版本使用最高精度
            DxfVersion::V2000
            | DxfVersion::V2004
            | DxfVersion::V2007
            | DxfVersion::V2010
            | DxfVersion::V2013
            | DxfVersion::V2018 => Self {
                point_tolerance: 1e-6,
                angle_tolerance: 0.1,
                chord_height_tolerance: 0.1,
                min_line_length: 0.001,
            },
            DxfVersion::Unknown => Self::default(),
        }
    }
}

impl DxfVersionFeatures {
    /// 根据版本创建特性配置
    #[allow(clippy::field_reassign_with_default)]
    pub fn for_version(version: DxfVersion) -> Self {
        let mut features = Self {
            tolerance_config: VersionToleranceConfig::for_version(version),
            ..Default::default()
        };

        // R12 基础实体
        let base_entities = [
            "LINE",
            "CIRCLE",
            "ARC",
            "POLYLINE",
            "LWPOLYLINE",
            "TEXT",
            "DIMENSION",
            "HATCH",
            "SOLID",
            "TRACE",
            "POINT",
        ];

        // R13/R14 新增
        let r13_entities = ["3DFACE", "BODY", "REGION", "MESH"];

        // 2000+ 新增
        let v2000_entities = [
            "SPLINE",
            "ELLIPSE",
            "MTEXT",
            "LEADER",
            "TOLERANCE",
            "VPOR T",
            "XLINE",
            "RAY",
            "MLINE",
        ];

        // 2007+ 新增
        let v2007_entities = [
            "CONE", "CYLINDER", "SPHERE", "TORUS", "BOX", "WEDGE", "PYRAMID", "SURFACE", "LOFT",
            "SWEEP",
        ];

        match version {
            DxfVersion::R12 => {
                features.supported_entities = base_entities.iter().map(|s| s.to_string()).collect();
                features.unsupported_entities = ["SPLINE", "ELLIPSE", "MTEXT", "3DFACE", "BODY"]
                    .iter()
                    .map(|s| s.to_string())
                    .collect();
                // R12 POLYLINE 需要特殊处理（旧格式）
                features
                    .special_handling
                    .insert("POLYLINE".to_string(), "r12_polyline_format".to_string());
            }
            DxfVersion::R13 | DxfVersion::R14 => {
                let mut entities: HashSet<String> =
                    base_entities.iter().map(|s| s.to_string()).collect();
                entities.extend(r13_entities.iter().map(|s| s.to_string()));
                features.supported_entities = entities;
                features.unsupported_entities = ["SPLINE", "ELLIPSE", "MTEXT"]
                    .iter()
                    .map(|s| s.to_string())
                    .collect();
            }
            DxfVersion::V2000 | DxfVersion::V2004 => {
                let mut entities: HashSet<String> =
                    base_entities.iter().map(|s| s.to_string()).collect();
                entities.extend(r13_entities.iter().map(|s| s.to_string()));
                entities.extend(v2000_entities.iter().map(|s| s.to_string()));
                features.supported_entities = entities;
            }
            DxfVersion::V2007 | DxfVersion::V2010 | DxfVersion::V2013 | DxfVersion::V2018 => {
                let mut entities: HashSet<String> =
                    base_entities.iter().map(|s| s.to_string()).collect();
                entities.extend(r13_entities.iter().map(|s| s.to_string()));
                entities.extend(v2000_entities.iter().map(|s| s.to_string()));
                entities.extend(v2007_entities.iter().map(|s| s.to_string()));
                features.supported_entities = entities;
                // NURBS 曲线需要特殊处理
                features
                    .special_handling
                    .insert("SPLINE".to_string(), "nurbs_discretization".to_string());
            }
            DxfVersion::Unknown => {
                // 未知版本：尝试支持所有常见实体
                features
                    .supported_entities
                    .extend(base_entities.iter().map(|s| s.to_string()));
                features
                    .supported_entities
                    .extend(r13_entities.iter().map(|s| s.to_string()));
                features
                    .supported_entities
                    .extend(v2000_entities.iter().map(|s| s.to_string()));
            }
        }

        features
    }

    /// 检查实体类型是否支持
    pub fn is_entity_supported(&self, entity_type: &str) -> bool {
        self.supported_entities
            .contains(entity_type.to_uppercase().as_str())
    }

    /// 获取特殊处理方式
    pub fn get_special_handling(&self, entity_type: &str) -> Option<&str> {
        self.special_handling
            .get(entity_type.to_uppercase().as_str())
            .map(|s| s.as_str())
    }
}

/// DXF 版本检测器
pub struct DxfVersionDetector {
    /// 检测到的版本
    detected_version: Option<DxfVersion>,
    /// 版本检测置信度（0.0-1.0）
    confidence: f64,
}

impl DxfVersionDetector {
    /// 创建新的版本检测器
    pub fn new() -> Self {
        Self {
            detected_version: None,
            confidence: 0.0,
        }
    }

    /// 从文件头检测 DXF 版本
    ///
    /// # 参数
    /// - `header_bytes`: 文件前 20 字节
    ///
    /// # 返回
    /// 检测到的版本和置信度
    pub fn detect_from_header(&mut self, header_bytes: &[u8]) -> (DxfVersion, f64) {
        if header_bytes.len() < 6 {
            self.detected_version = Some(DxfVersion::Unknown);
            self.confidence = 0.0;
            return (DxfVersion::Unknown, 0.0);
        }

        // 尝试解析版本代码
        let version_str = String::from_utf8_lossy(&header_bytes[0..6]);
        let version = DxfVersion::from_code(&version_str);

        if version != DxfVersion::Unknown {
            self.detected_version = Some(version);
            self.confidence = 1.0;
        } else {
            // 尝试更宽松的匹配
            self.detected_version = Some(DxfVersion::Unknown);
            self.confidence = 0.3;
        }

        (version, self.confidence)
    }

    /// 从 Drawing 对象获取版本
    pub fn detect_from_drawing(&mut self, drawing: &dxf::Drawing) -> DxfVersion {
        // 尝试从 header 获取版本
        let version_code = drawing.header.version.to_string();
        let version = DxfVersion::from_code(&version_code);
        if version != DxfVersion::Unknown {
            self.detected_version = Some(version);
            self.confidence = 1.0;
            return version;
        }

        // 默认返回未知版本
        self.detected_version = Some(DxfVersion::Unknown);
        self.confidence = 0.5;
        DxfVersion::Unknown
    }

    /// 从实体类型推断版本（简化版）
    #[allow(dead_code)] // 预留用于未来 DXF 版本自动检测优化
    fn infer_version_from_entities(&self, _drawing: &dxf::Drawing) -> DxfVersion {
        // 简化版本：默认返回 V2000
        DxfVersion::V2000
    }

    /// 获取检测到的版本
    pub fn detected_version(&self) -> Option<DxfVersion> {
        self.detected_version
    }

    /// 获取置信度
    pub fn confidence(&self) -> f64 {
        self.confidence
    }
}

impl Default for DxfVersionDetector {
    fn default() -> Self {
        Self::new()
    }
}

/// 版本兼容性报告
#[derive(Debug, Clone)]
pub struct VersionCompatibilityReport {
    /// 检测到的版本
    pub detected_version: DxfVersion,
    /// 兼容性评分（0-100）
    pub compatibility_score: u8,
    /// 不支持的实体列表
    pub unsupported_entities: Vec<String>,
    /// 警告信息
    pub warnings: Vec<String>,
    /// 推荐的处理策略
    pub recommended_strategy: DxfVersionStrategy,
}

impl Default for VersionCompatibilityReport {
    fn default() -> Self {
        Self {
            detected_version: DxfVersion::Unknown,
            compatibility_score: 0,
            unsupported_entities: Vec::new(),
            warnings: Vec::new(),
            recommended_strategy: DxfVersionStrategy::Fallback,
        }
    }
}

impl VersionCompatibilityReport {
    /// 创建新的兼容性报告
    pub fn new(version: DxfVersion) -> Self {
        Self {
            detected_version: version,
            compatibility_score: version.compatibility_score(),
            unsupported_entities: Vec::new(),
            warnings: Vec::new(),
            recommended_strategy: version.recommended_strategy(),
        }
    }

    /// 添加警告
    pub fn add_warning(&mut self, warning: impl Into<String>) {
        self.warnings.push(warning.into());
    }

    /// 添加不支持的实体
    pub fn add_unsupported_entity(&mut self, entity: impl Into<String>) {
        let entity = entity.into();
        if !self.unsupported_entities.contains(&entity) {
            self.unsupported_entities.push(entity);
        }
    }

    /// 是否完全兼容
    pub fn is_fully_compatible(&self) -> bool {
        self.compatibility_score >= 90 && self.unsupported_entities.is_empty()
    }

    /// 是否需要特殊处理
    pub fn requires_special_handling(&self) -> bool {
        matches!(
            self.detected_version,
            DxfVersion::R12 | DxfVersion::R13 | DxfVersion::R14
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version_from_code() {
        assert_eq!(DxfVersion::from_code("AC1009"), DxfVersion::R12);
        assert_eq!(DxfVersion::from_code("AC1015"), DxfVersion::V2000);
        assert_eq!(DxfVersion::from_code("AC1032"), DxfVersion::V2018);
        assert_eq!(DxfVersion::from_code("INVALID"), DxfVersion::Unknown);
    }

    #[test]
    fn test_version_to_code() {
        assert_eq!(DxfVersion::R12.to_code(), "AC1009");
        assert_eq!(DxfVersion::V2018.to_code(), "AC1032");
        assert_eq!(DxfVersion::Unknown.to_code(), "UNKNOWN");
    }

    #[test]
    fn test_version_features() {
        // R12 不支持 SPLINE
        let r12_features = DxfVersionFeatures::for_version(DxfVersion::R12);
        assert!(!r12_features.is_entity_supported("SPLINE"));
        assert!(r12_features.is_entity_supported("LINE"));

        // V2007 支持 SPLINE
        let v2007_features = DxfVersionFeatures::for_version(DxfVersion::V2007);
        assert!(v2007_features.is_entity_supported("SPLINE"));
    }

    #[test]
    fn test_version_tolerance_config() {
        // R12 容差较大
        let r12_config = VersionToleranceConfig::for_version(DxfVersion::R12);
        assert!(r12_config.point_tolerance > 1e-5);

        // V2018 容差较小
        let v2018_config = VersionToleranceConfig::for_version(DxfVersion::V2018);
        assert!(v2018_config.point_tolerance < 1e-5);
    }

    #[test]
    fn test_compatibility_score() {
        assert_eq!(DxfVersion::V2018.compatibility_score(), 100);
        assert_eq!(DxfVersion::R12.compatibility_score(), 50);
        assert_eq!(DxfVersion::Unknown.compatibility_score(), 0);
    }

    #[test]
    fn test_version_strategy() {
        assert!(DxfVersionStrategy::LegacyR12.is_strict());
        assert!(!DxfVersionStrategy::ModernV2007Plus.is_strict());
    }
}
