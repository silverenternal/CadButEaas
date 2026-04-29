//! 统一配置管理
//!
//! 通过 TOML 配置文件管理所有服务的参数
//!
//! ## 使用示例
//! ```no_run
//! use config::CadConfig;
//!
//! // 从文件加载配置
//! let config = CadConfig::from_file("cad_config.toml").unwrap();
//!
//! // 或使用默认配置
//! let config = CadConfig::default();
//!
//! // 或使用预设配置
//! let config = CadConfig::from_profile("architectural").unwrap();
//! ```

use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

/// CAD 处理系统统一配置
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CadConfig {
    /// 预设配置名称（可选）
    #[serde(skip)]
    pub profile_name: Option<String>,
    /// 解析器配置
    pub parser: ParserConfig,
    /// 拓扑配置
    pub topology: TopoConfig,
    /// 验证器配置
    pub validator: ValidatorConfig,
    /// 导出配置
    pub export: ExportConfig,
    /// 光栅图纸增强解析配置
    #[serde(default)]
    pub raster: RasterConfig,
}

/// DXF 解析器配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParserConfig {
    /// DXF 特定配置
    pub dxf: DxfConfig,
    /// PDF 特定配置
    pub pdf: PdfConfig,
}

/// DXF 配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DxfConfig {
    /// 图层白名单
    pub layer_whitelist: Option<Vec<String>>,
    /// 实体类型白名单
    pub entity_whitelist: Option<Vec<String>>,
    /// 颜色白名单 (ACI 索引 1-255)
    pub color_whitelist: Option<Vec<i16>>,
    /// 线宽白名单 (DXF 枚举值：-3..=-1=BYLAYER/BYBLOCK/DEFAULT, 0=0.00mm, 7=0.25mm, 11=0.50mm, 21=2.11mm)
    pub lineweight_whitelist: Option<Vec<i16>>,
    /// ARC 离散化容差 (mm)
    pub arc_tolerance_mm: f64,
    /// SPLINE 离散化容差 (mm)
    pub spline_tolerance_mm: f64,
    /// 忽略文本实体
    pub ignore_text: bool,
    /// 忽略标注实体
    pub ignore_dimensions: bool,
    /// 忽略填充实体
    pub ignore_hatch: bool,
}

/// PDF 配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PdfConfig {
    /// 矢量化容差 (像素)
    pub vectorize_tolerance_px: f64,
    /// 边缘检测阈值 (0.0-1.0)
    pub edge_threshold: f64,
    /// 最小线段长度 (像素)
    pub min_line_length_px: f64,
    /// 二值化阈值 (0-255) - P11 锐评 v2.0 修复：移除硬编码
    #[serde(default = "default_threshold")]
    pub threshold: u8,
    /// 最大图像像素数限制（默认 30,000,000）
    #[serde(default = "default_max_pixels")]
    pub max_pixels: usize,
}

/// 光栅增强解析配置。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RasterConfig {
    /// auto/clean_line_art/scanned_plan/photo_perspective/hand_sketch/low_contrast
    #[serde(default = "default_raster_strategy")]
    pub strategy: String,
    #[serde(default)]
    pub dpi_override: Option<(f64, f64)>,
    #[serde(default)]
    pub debug_artifacts: bool,
    #[serde(default = "default_raster_max_retries")]
    pub max_retries: usize,
    #[serde(default = "default_semantic_mode")]
    pub semantic_mode: String,
    #[serde(default = "default_ocr_backend")]
    pub ocr_backend: String,
}

fn default_raster_strategy() -> String {
    "auto".to_string()
}

fn default_raster_max_retries() -> usize {
    3
}

fn default_semantic_mode() -> String {
    "rule".to_string()
}

fn default_ocr_backend() -> String {
    "heuristic".to_string()
}

fn default_threshold() -> u8 {
    128
}

fn default_max_pixels() -> usize {
    30_000_000
}

/// 拓扑配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopoConfig {
    /// 端点吸附容差 (mm)
    pub snap_tolerance_mm: f64,
    /// 最小线段长度 (mm)
    pub min_line_length_mm: f64,
    /// 共线合并角度容差 (度)
    pub merge_angle_tolerance_deg: f64,
    /// 最大间隙桥接长度 (mm)
    pub max_gap_bridge_length_mm: f64,
    /// P11 新增：拓扑构建算法
    /// - "dfs": DFS 方案（默认，向后兼容）
    /// - "halfedge": Halfedge 方案（推荐，支持嵌套孔洞）
    #[serde(default = "default_algorithm")]
    pub algorithm: String,
    /// 跳过交点检测（P11 性能优化）
    /// true = 跳过交点检测和切分，适用于已清理的 DXF 文件
    /// false = 执行完整的交点检测（默认，处理复杂图纸）
    #[serde(default)]
    pub skip_intersection_check: bool,
    /// 启用并行处理（P11 锐评落实）
    /// true = 大场景自动启用并行端点吸附和交点检测
    /// false = 使用串行处理（默认，兼容旧流程）
    #[serde(default)]
    pub enable_parallel: bool,
    /// 并行处理阈值（P11 锐评落实）
    /// 当线段数量超过此阈值时自动启用并行处理
    #[serde(default = "default_parallel_threshold")]
    pub parallel_threshold: usize,
}

fn default_algorithm() -> String {
    "dfs".to_string()
}

fn default_parallel_threshold() -> usize {
    1000
}

/// 验证器配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidatorConfig {
    /// 闭合性检查容差 (mm)
    pub closure_tolerance_mm: f64,
    /// 最小面积 (m²)
    pub min_area_m2: f64,
    /// 最短边检查 (mm)
    pub min_edge_length_mm: f64,
    /// 最小角度检查 (度)
    pub min_angle_deg: f64,
}

/// 导出配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportConfig {
    /// 导出格式 (json/bincode)
    pub format: String,
    /// JSON 缩进空格数
    pub json_indent: usize,
    /// 导出前自动验证
    pub auto_validate: bool,
}

#[allow(clippy::derivable_impls)]
impl Default for ParserConfig {
    fn default() -> Self {
        Self {
            dxf: DxfConfig::default(),
            pdf: PdfConfig::default(),
        }
    }
}

#[allow(clippy::derivable_impls)]
impl Default for DxfConfig {
    fn default() -> Self {
        Self {
            layer_whitelist: None,
            entity_whitelist: None,
            color_whitelist: None,
            lineweight_whitelist: None,
            arc_tolerance_mm: 0.1,
            spline_tolerance_mm: 0.1,
            ignore_text: true,
            ignore_dimensions: true,
            ignore_hatch: true,
        }
    }
}

#[allow(clippy::derivable_impls)]
impl Default for PdfConfig {
    fn default() -> Self {
        Self {
            vectorize_tolerance_px: 1.0,
            edge_threshold: 0.1,
            min_line_length_px: 5.0,
            threshold: 128,
            max_pixels: 30_000_000,
        }
    }
}

impl Default for RasterConfig {
    fn default() -> Self {
        Self {
            strategy: default_raster_strategy(),
            dpi_override: None,
            debug_artifacts: false,
            max_retries: 3,
            semantic_mode: default_semantic_mode(),
            ocr_backend: default_ocr_backend(),
        }
    }
}

#[allow(clippy::derivable_impls)]
impl Default for TopoConfig {
    fn default() -> Self {
        Self {
            snap_tolerance_mm: 0.5,
            min_line_length_mm: 1.0,
            merge_angle_tolerance_deg: 5.0,
            max_gap_bridge_length_mm: 2.0,
            algorithm: "dfs".to_string(),
            skip_intersection_check: false,
            enable_parallel: false,
            parallel_threshold: 1000,
        }
    }
}

#[allow(clippy::derivable_impls)]
impl Default for ValidatorConfig {
    fn default() -> Self {
        Self {
            closure_tolerance_mm: 0.5,
            min_area_m2: 1.0,
            min_edge_length_mm: 10.0,
            min_angle_deg: 15.0,
        }
    }
}

#[allow(clippy::derivable_impls)]
impl Default for ExportConfig {
    fn default() -> Self {
        Self {
            format: "json".to_string(),
            json_indent: 2,
            auto_validate: true,
        }
    }
}

impl CadConfig {
    /// 从 TOML 文件加载配置
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, ConfigError> {
        let path = path.as_ref();
        let content = fs::read_to_string(path).map_err(|e| ConfigError::FileReadError {
            path: path.to_path_buf(),
            source: e,
        })?;

        let config: CadConfig = toml::from_str(&content).map_err(|e| ConfigError::ParseError {
            path: path.to_path_buf(),
            source: Box::new(e),
        })?;

        Ok(config)
    }

    /// 保存到 TOML 文件
    pub fn save_to_file(&self, path: impl AsRef<Path>) -> Result<(), ConfigError> {
        let path = path.as_ref();
        let content = toml::to_string_pretty(self).map_err(|e| ConfigError::SerializeError {
            source: Box::new(e),
        })?;

        fs::write(path, content).map_err(|e| ConfigError::FileWriteError {
            path: path.to_path_buf(),
            source: e,
        })?;

        Ok(())
    }

    /// 生成默认配置文件到指定路径
    pub fn generate_default_file(path: impl AsRef<Path>) -> Result<(), ConfigError> {
        let config = Self::default();
        config.save_to_file(path)
    }

    /// 验证配置的有效性
    pub fn validate(&self) -> Result<(), ConfigError> {
        // 验证容差值合理性
        if self.parser.dxf.arc_tolerance_mm <= 0.0 {
            return Err(ConfigError::InvalidValue {
                field: "parser.dxf.arc_tolerance_mm".to_string(),
                reason: "必须大于 0".to_string(),
            });
        }

        if self.topology.snap_tolerance_mm <= 0.0 {
            return Err(ConfigError::InvalidValue {
                field: "topology.snap_tolerance_mm".to_string(),
                reason: "必须大于 0".to_string(),
            });
        }

        if self.parser.pdf.edge_threshold < 0.0 || self.parser.pdf.edge_threshold > 1.0 {
            return Err(ConfigError::InvalidValue {
                field: "parser.pdf.edge_threshold".to_string(),
                reason: "必须在 0.0-1.0 范围内".to_string(),
            });
        }

        Ok(())
    }

    /// 从预设配置加载
    ///
    /// 支持的预设：
    /// - `architectural`: 建筑图纸预设
    /// - `mechanical`: 机械图纸预设
    /// - `scanned`: 扫描图纸预设（仅适用于线条清晰的图纸）
    /// - `photo_sketch`: 照片/手绘草图预设（复杂光栅图片，强预处理）
    /// - `quick`: 快速原型预设
    ///
    /// ## 注意
    ///
    /// 此方法使用硬编码的预设值。如需从配置文件加载预设，请使用：
    /// - `CadConfig::from_profile_file(profile_name)` - 自动搜索配置文件
    /// - `CadConfig::from_profile_file_path(profile_name, path)` - 指定配置文件路径
    pub fn from_profile(profile_name: &str) -> Result<Self, ConfigError> {
        let mut config = match profile_name.to_lowercase().as_str() {
            "architectural" => Self::architectural_profile(),
            "mechanical" => Self::mechanical_profile(),
            "scanned" => Self::scanned_profile(),
            "photo_sketch" => Self::photo_sketch_profile(),
            "raster_clean" => Self::raster_profile("raster_clean", "clean_line_art", 2),
            "raster_scan" => Self::raster_profile("raster_scan", "scanned_plan", 3),
            "raster_photo" => Self::raster_profile("raster_photo", "photo_perspective", 3),
            "raster_sketch" => Self::raster_profile("raster_sketch", "hand_sketch", 3),
            "raster_semantic" => {
                let mut cfg = Self::raster_profile("raster_semantic", "auto", 3);
                cfg.raster.semantic_mode = "semantic".to_string();
                cfg
            }
            "quick" => Self::quick_profile(),
            _ => {
                return Err(ConfigError::InvalidValue {
                    field: "profile".to_string(),
                    reason: format!(
                        "未知的预设配置 '{}'，支持的预设：architectural, mechanical, scanned, photo_sketch, quick",
                        profile_name
                    ),
                });
            }
        };

        config.profile_name = Some(profile_name.to_string());
        Ok(config)
    }

    fn raster_profile(_name: &str, strategy: &str, max_retries: usize) -> Self {
        let mut config = Self::scanned_profile();
        config.parser.dxf.ignore_text = false;
        config.parser.dxf.ignore_dimensions = false;
        config.raster = RasterConfig {
            strategy: strategy.to_string(),
            max_retries,
            ..Default::default()
        };
        config
    }

    /// 建筑图纸预设
    fn architectural_profile() -> Self {
        Self {
            parser: ParserConfig {
                dxf: DxfConfig {
                    layer_whitelist: Some(vec![
                        "WALL".to_string(),
                        "WALL-*".to_string(),
                        "A-WALL".to_string(),
                        "墙".to_string(),
                        "墙体".to_string(),
                        "QIANG".to_string(),
                    ]),
                    entity_whitelist: Some(vec![
                        "LINE".to_string(),
                        "LWPOLYLINE".to_string(),
                        "ARC".to_string(),
                        "CIRCLE".to_string(),
                        "SPLINE".to_string(),
                    ]),
                    arc_tolerance_mm: 0.1,
                    spline_tolerance_mm: 0.1,
                    ignore_text: true,
                    ignore_dimensions: true,
                    ignore_hatch: true,
                    ..Default::default()
                },
                pdf: PdfConfig {
                    vectorize_tolerance_px: 1.0,
                    edge_threshold: 0.1,
                    min_line_length_px: 5.0,
                    threshold: 128,
                    max_pixels: 30_000_000,
                },
            },
            topology: TopoConfig {
                snap_tolerance_mm: 0.5,
                min_line_length_mm: 1.0,
                merge_angle_tolerance_deg: 5.0,
                max_gap_bridge_length_mm: 2.0,
                algorithm: "dfs".to_string(),
                skip_intersection_check: false,
                enable_parallel: false,
                parallel_threshold: 1000,
            },
            validator: ValidatorConfig {
                closure_tolerance_mm: 0.5,
                min_area_m2: 1.0,
                min_edge_length_mm: 10.0,
                min_angle_deg: 15.0,
            },
            export: ExportConfig {
                format: "json".to_string(),
                json_indent: 2,
                auto_validate: true,
            },
            raster: RasterConfig {
                strategy: "clean_line_art".to_string(),
                ..Default::default()
            },
            profile_name: None,
        }
    }

    /// 机械图纸预设
    fn mechanical_profile() -> Self {
        Self {
            parser: ParserConfig {
                dxf: DxfConfig {
                    layer_whitelist: Some(vec![
                        "DIM".to_string(),
                        "CENTER".to_string(),
                        "HATCH".to_string(),
                        "OBJECT".to_string(),
                        "标注".to_string(),
                        "中心线".to_string(),
                        "剖面线".to_string(),
                    ]),
                    entity_whitelist: Some(vec![
                        "LINE".to_string(),
                        "LWPOLYLINE".to_string(),
                        "ARC".to_string(),
                        "CIRCLE".to_string(),
                        "SPLINE".to_string(),
                        "ELLIPSE".to_string(),
                    ]),
                    arc_tolerance_mm: 0.01,
                    spline_tolerance_mm: 0.01,
                    ignore_text: false,
                    ignore_dimensions: false,
                    ignore_hatch: false,
                    ..Default::default()
                },
                pdf: PdfConfig {
                    vectorize_tolerance_px: 0.5,
                    edge_threshold: 0.08,
                    min_line_length_px: 3.0,
                    threshold: 128,
                    max_pixels: 30_000_000,
                },
            },
            topology: TopoConfig {
                snap_tolerance_mm: 0.1,
                min_line_length_mm: 0.5,
                merge_angle_tolerance_deg: 2.0,
                max_gap_bridge_length_mm: 0.5,
                algorithm: "dfs".to_string(),
                skip_intersection_check: false,
                enable_parallel: false,
                parallel_threshold: 1000,
            },
            validator: ValidatorConfig {
                closure_tolerance_mm: 0.1,
                min_area_m2: 0.01,
                min_edge_length_mm: 1.0,
                min_angle_deg: 5.0,
            },
            export: ExportConfig {
                format: "bincode".to_string(),
                json_indent: 2,
                auto_validate: true,
            },
            raster: RasterConfig {
                strategy: "clean_line_art".to_string(),
                semantic_mode: "semantic".to_string(),
                ..Default::default()
            },
            profile_name: None,
        }
    }

    /// 扫描图纸预设（仅适用于线条清晰的图纸）
    fn scanned_profile() -> Self {
        Self {
            parser: ParserConfig {
                dxf: DxfConfig {
                    layer_whitelist: None,
                    entity_whitelist: Some(vec!["LINE".to_string(), "LWPOLYLINE".to_string()]),
                    arc_tolerance_mm: 0.5,
                    spline_tolerance_mm: 0.5,
                    ignore_text: true,
                    ignore_dimensions: true,
                    ignore_hatch: true,
                    ..Default::default()
                },
                pdf: PdfConfig {
                    vectorize_tolerance_px: 2.0,
                    edge_threshold: 0.15,
                    min_line_length_px: 8.0,
                    threshold: 128,
                    max_pixels: 30_000_000,
                },
            },
            topology: TopoConfig {
                snap_tolerance_mm: 2.0,
                min_line_length_mm: 3.0,
                merge_angle_tolerance_deg: 10.0,
                max_gap_bridge_length_mm: 5.0,
                algorithm: "dfs".to_string(),
                skip_intersection_check: false,
                enable_parallel: false,
                parallel_threshold: 1000,
            },
            validator: ValidatorConfig {
                closure_tolerance_mm: 2.0,
                min_area_m2: 2.0,
                min_edge_length_mm: 20.0,
                min_angle_deg: 30.0,
            },
            export: ExportConfig {
                format: "json".to_string(),
                json_indent: 2,
                auto_validate: true,
            },
            raster: RasterConfig {
                strategy: "scanned_plan".to_string(),
                max_retries: 3,
                ..Default::default()
            },
            profile_name: None,
        }
    }

    /// 照片/手绘草图预设（复杂光栅图片，强预处理）
    fn photo_sketch_profile() -> Self {
        Self {
            parser: ParserConfig {
                dxf: DxfConfig {
                    layer_whitelist: None,
                    entity_whitelist: Some(vec!["LINE".to_string(), "LWPOLYLINE".to_string()]),
                    arc_tolerance_mm: 0.5,
                    spline_tolerance_mm: 0.5,
                    ignore_text: true,
                    ignore_dimensions: true,
                    ignore_hatch: true,
                    ..Default::default()
                },
                pdf: PdfConfig {
                    vectorize_tolerance_px: 2.0,
                    edge_threshold: 0.12,
                    min_line_length_px: 3.0,
                    threshold: 128,
                    max_pixels: 30_000_000,
                },
            },
            topology: TopoConfig {
                snap_tolerance_mm: 2.0,
                min_line_length_mm: 3.0,
                merge_angle_tolerance_deg: 10.0,
                max_gap_bridge_length_mm: 5.0,
                algorithm: "halfedge".to_string(),
                skip_intersection_check: false,
                enable_parallel: true,
                parallel_threshold: 2000,
            },
            validator: ValidatorConfig {
                closure_tolerance_mm: 2.0,
                min_area_m2: 2.0,
                min_edge_length_mm: 20.0,
                min_angle_deg: 30.0,
            },
            export: ExportConfig {
                format: "json".to_string(),
                json_indent: 2,
                auto_validate: true,
            },
            raster: RasterConfig {
                strategy: "hand_sketch".to_string(),
                max_retries: 3,
                ..Default::default()
            },
            profile_name: None,
        }
    }

    /// 快速原型预设
    fn quick_profile() -> Self {
        Self {
            parser: ParserConfig {
                dxf: DxfConfig {
                    layer_whitelist: None,
                    entity_whitelist: Some(vec!["LINE".to_string(), "LWPOLYLINE".to_string()]),
                    arc_tolerance_mm: 1.0,
                    spline_tolerance_mm: 1.0,
                    ignore_text: true,
                    ignore_dimensions: true,
                    ignore_hatch: true,
                    ..Default::default()
                },
                pdf: PdfConfig::default(),
            },
            topology: TopoConfig {
                snap_tolerance_mm: 1.0,
                min_line_length_mm: 5.0,
                merge_angle_tolerance_deg: 15.0,
                max_gap_bridge_length_mm: 1.0,
                algorithm: "dfs".to_string(),
                skip_intersection_check: false,
                enable_parallel: false,
                parallel_threshold: 1000,
            },
            validator: ValidatorConfig {
                closure_tolerance_mm: 1.0,
                min_area_m2: 0.5,
                min_edge_length_mm: 5.0,
                min_angle_deg: 10.0,
            },
            export: ExportConfig {
                format: "json".to_string(),
                json_indent: 0,
                auto_validate: false,
            },
            raster: RasterConfig {
                strategy: "auto".to_string(),
                max_retries: 1,
                ..Default::default()
            },
            profile_name: None,
        }
    }

    /// 从配置文件加载预设配置
    ///
    /// 尝试从以下路径加载 cad_config.profiles.toml：
    /// 1. 当前工作目录
    /// 2. CARGO_MANIFEST_DIR 指向的目录
    /// 3. 用户主目录下的 .cad 目录
    pub fn from_profile_file(profile_name: &str) -> Result<Self, ConfigError> {
        use std::env;

        // 尝试从多个路径加载配置文件
        let mut search_paths: Vec<PathBuf> = Vec::new();

        // 当前工作目录
        search_paths.push(PathBuf::from("cad_config.profiles.toml"));

        // 项目根目录（开发环境）
        if let Ok(manifest_dir) = env::var("CARGO_MANIFEST_DIR") {
            search_paths.push(PathBuf::from(manifest_dir).join("../../cad_config.profiles.toml"));
        }

        // 用户主目录
        if let Ok(home_dir) = env::var("HOME") {
            search_paths.push(PathBuf::from(home_dir).join(".cad/cad_config.profiles.toml"));
        }

        for path in search_paths {
            if path.exists() {
                return Self::from_profile_file_path(profile_name, &path);
            }
        }

        // 如果都找不到，回退到硬编码预设
        tracing::warn!("未找到 cad_config.profiles.toml，使用硬编码预设配置");
        Self::from_profile(profile_name)
    }

    /// 从指定路径的配置文件加载预设配置
    ///
    /// # P11 锐评落实
    ///
    /// 原文档指出：嵌套获取逻辑复杂容易出错。
    /// 修复方案：使用 serde 直接反序列化整个配置结构。
    pub fn from_profile_file_path(profile_name: &str, path: &Path) -> Result<Self, ConfigError> {
        let content = fs::read_to_string(path).map_err(|e| ConfigError::FileReadError {
            path: path.to_path_buf(),
            source: e,
        })?;

        // P11 锐评落实：使用 serde 直接反序列化，而非手动嵌套获取
        // 配置文件格式：
        // [profile.architectural.parser.dxf]
        // [profile.architectural.topology]
        // ...
        // 直接反序列化为 CadConfig 结构

        // 首先解析为 TOML Value
        let config_value: toml::Value =
            toml::from_str(&content).map_err(|e| ConfigError::ParseError {
                path: path.to_path_buf(),
                source: Box::new(e),
            })?;

        // 从 [profile.xxx] 表中获取预设配置
        let profile_value = config_value
            .get("profile")
            .and_then(|p| p.get(profile_name.to_lowercase()))
            .ok_or_else(|| ConfigError::InvalidValue {
                field: "profile".to_string(),
                reason: format!(
                    "配置文件 '{}' 中未找到预设配置 '{}'",
                    path.display(),
                    profile_name
                ),
            })?;

        // 使用 serde 直接反序列化为 CadConfig
        // 这比手动嵌套获取更简洁、更可靠
        let config: Self =
            toml::Value::try_into(profile_value.clone()).map_err(|e: toml::de::Error| {
                ConfigError::ParseError {
                    path: path.to_path_buf(),
                    source: Box::new(e),
                }
            })?;

        Ok(config)
    }
}

/// 配置错误类型
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("读取配置文件失败 {path:?}: {source}")]
    FileReadError {
        path: PathBuf,
        source: std::io::Error,
    },

    #[error("写入配置文件失败 {path:?}: {source}")]
    FileWriteError {
        path: PathBuf,
        source: std::io::Error,
    },

    #[error("解析配置文件失败 {path:?}: {source}")]
    ParseError {
        path: PathBuf,
        source: Box<toml::de::Error>,
    },

    #[error("序列化配置失败：{source}")]
    SerializeError { source: Box<toml::ser::Error> },

    #[error("配置值无效：字段 '{field}' - {reason}")]
    InvalidValue { field: String, reason: String },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = CadConfig::default();

        assert_eq!(config.parser.dxf.arc_tolerance_mm, 0.1);
        assert_eq!(config.topology.snap_tolerance_mm, 0.5);
        assert_eq!(config.validator.closure_tolerance_mm, 0.5);
        assert_eq!(config.export.format, "json");
    }

    #[test]
    fn test_config_validation() {
        let config = CadConfig::default();
        assert!(config.validate().is_ok());

        let mut invalid_config = config.clone();
        invalid_config.parser.dxf.arc_tolerance_mm = 0.0;
        assert!(invalid_config.validate().is_err());

        let mut invalid_config = config.clone();
        invalid_config.parser.pdf.edge_threshold = 1.5;
        assert!(invalid_config.validate().is_err());
    }

    #[test]
    fn test_config_serialization() {
        let config = CadConfig::default();
        let toml_str = toml::to_string(&config).unwrap();

        let parsed: CadConfig = toml::from_str(&toml_str).unwrap();
        assert_eq!(
            parsed.parser.dxf.arc_tolerance_mm,
            config.parser.dxf.arc_tolerance_mm
        );
    }

    #[test]
    fn test_config_roundtrip() {
        let config = CadConfig::default();
        let temp_path = std::env::temp_dir().join("cad_config_test.toml");

        config.save_to_file(&temp_path).unwrap();
        let loaded = CadConfig::from_file(&temp_path).unwrap();

        assert_eq!(
            loaded.parser.dxf.arc_tolerance_mm,
            config.parser.dxf.arc_tolerance_mm
        );

        fs::remove_file(temp_path).ok();
    }

    #[test]
    fn test_profile_architectural() {
        let config = CadConfig::from_profile("architectural").unwrap();
        assert_eq!(config.profile_name, Some("architectural".to_string()));
        assert_eq!(config.topology.snap_tolerance_mm, 0.5);
        assert_eq!(config.parser.dxf.arc_tolerance_mm, 0.1);
        assert!(config.parser.dxf.layer_whitelist.is_some());
    }

    #[test]
    fn test_profile_mechanical() {
        let config = CadConfig::from_profile("mechanical").unwrap();
        assert_eq!(config.profile_name, Some("mechanical".to_string()));
        assert_eq!(config.topology.snap_tolerance_mm, 0.1);
        assert_eq!(config.parser.dxf.arc_tolerance_mm, 0.01);
        assert_eq!(config.export.format, "bincode");
    }

    #[test]
    fn test_profile_scanned() {
        let config = CadConfig::from_profile("scanned").unwrap();
        assert_eq!(config.profile_name, Some("scanned".to_string()));
        assert_eq!(config.topology.snap_tolerance_mm, 2.0);
        assert_eq!(config.parser.pdf.edge_threshold, 0.15);
    }

    #[test]
    fn test_profile_photo_sketch() {
        let config = CadConfig::from_profile("photo_sketch").unwrap();
        assert_eq!(config.profile_name, Some("photo_sketch".to_string()));
        assert_eq!(config.topology.algorithm, "halfedge");
        assert!(config.topology.enable_parallel);
        assert_eq!(config.parser.pdf.min_line_length_px, 3.0);
    }

    #[test]
    fn test_profile_quick() {
        let config = CadConfig::from_profile("quick").unwrap();
        assert_eq!(config.profile_name, Some("quick".to_string()));
        assert_eq!(config.topology.snap_tolerance_mm, 1.0);
        assert!(!config.export.auto_validate);
    }

    #[test]
    fn test_profile_invalid() {
        let result = CadConfig::from_profile("invalid_profile");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("未知的预设配置"));
    }

    #[test]
    fn test_profile_case_insensitive() {
        let config1 = CadConfig::from_profile("Architectural").unwrap();
        let config2 = CadConfig::from_profile("ARCHITECTURAL").unwrap();
        let config3 = CadConfig::from_profile("architectural").unwrap();

        // 所有预设应该相同
        assert_eq!(
            config1.topology.snap_tolerance_mm,
            config2.topology.snap_tolerance_mm
        );
        assert_eq!(
            config1.topology.snap_tolerance_mm,
            config3.topology.snap_tolerance_mm
        );
    }

    #[test]
    fn test_from_profile_file_path() {
        // 创建一个临时配置文件用于测试
        let temp_dir = std::env::temp_dir();
        let config_path = temp_dir.join("cad_config_test_profiles.toml");

        let config_content = r#"
[profile.test_profile]
description = "测试预设配置"

[profile.test_profile.parser.dxf]
layer_whitelist = ["WALL", "DOOR"]
arc_tolerance_mm = 0.25
spline_tolerance_mm = 0.5
ignore_text = false
ignore_dimensions = true
ignore_hatch = false

[profile.test_profile.parser.pdf]
vectorize_tolerance_px = 1.5
edge_threshold = 0.12
min_line_length_px = 5.0
threshold = 128

[profile.test_profile.topology]
snap_tolerance_mm = 0.75
min_line_length_mm = 2.0
merge_angle_tolerance_deg = 7.5
max_gap_bridge_length_mm = 3.0

[profile.test_profile.validator]
closure_tolerance_mm = 0.75
min_area_m2 = 1.5
min_edge_length_mm = 15.0
min_angle_deg = 20.0

[profile.test_profile.export]
format = "bincode"
json_indent = 4
auto_validate = false
"#;

        fs::write(&config_path, config_content).unwrap();

        // 测试从配置文件加载预设
        let config = CadConfig::from_profile_file_path("test_profile", &config_path).unwrap();

        // 验证 parser.dxf 配置
        assert_eq!(
            config.parser.dxf.layer_whitelist,
            Some(vec!["WALL".to_string(), "DOOR".to_string()])
        );
        assert_eq!(config.parser.dxf.arc_tolerance_mm, 0.25);
        assert!(!config.parser.dxf.ignore_text);

        // 验证 parser.pdf 配置
        assert_eq!(config.parser.pdf.vectorize_tolerance_px, 1.5);
        assert_eq!(config.parser.pdf.edge_threshold, 0.12);

        // 验证 topology 配置
        assert_eq!(config.topology.snap_tolerance_mm, 0.75);
        assert_eq!(config.topology.min_line_length_mm, 2.0);
        assert_eq!(config.topology.merge_angle_tolerance_deg, 7.5);
        assert_eq!(config.topology.max_gap_bridge_length_mm, 3.0);

        // 验证 validator 配置
        assert_eq!(config.validator.closure_tolerance_mm, 0.75);
        assert_eq!(config.validator.min_area_m2, 1.5);
        assert_eq!(config.validator.min_edge_length_mm, 15.0);
        assert_eq!(config.validator.min_angle_deg, 20.0);

        // 验证 export 配置
        assert_eq!(config.export.format, "bincode");
        assert_eq!(config.export.json_indent, 4);
        assert!(!config.export.auto_validate);

        // 清理临时文件
        fs::remove_file(config_path).ok();
    }

    #[test]
    fn test_from_profile_file_path_missing_profile() {
        // 创建一个临时配置文件用于测试
        let temp_dir = std::env::temp_dir();
        let config_path = temp_dir.join("cad_config_test_missing.toml");

        let config_content = r#"
[profile.existing_profile]
description = "测试预设配置"

[profile.existing_profile.parser.dxf]
arc_tolerance_mm = 0.1
"#;

        fs::write(&config_path, config_content).unwrap();

        // 测试加载不存在的预设配置
        let result = CadConfig::from_profile_file_path("nonexistent_profile", &config_path);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("未找到预设配置"));

        // 清理临时文件
        fs::remove_file(config_path).ok();
    }
}
