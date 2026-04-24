//! 声学分析类型定义
//!
//! 用于 AcousticService 的 Input/Output

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};

use common_types::geometry::Point2;
use common_types::scene::SurfaceId;

// ============================================================================
// 输入类型
// ============================================================================

/// 声学分析输入
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct AcousticInput {
    /// 场景状态（包含几何和材料信息）
    pub scene: common_types::scene::SceneState,
    /// 分析请求
    pub request: AcousticRequest,
}

/// 声学分析请求
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "type", rename_all = "SCREAMING_SNAKE_CASE")]
pub enum AcousticRequest {
    /// 选区材料统计
    SelectionMaterialStats {
        /// 选区边界
        boundary: SelectionBoundary,
        /// 选区模式
        mode: SelectionMode,
    },
    /// 房间级混响时间计算
    RoomReverberation {
        /// 房间 ID（必须是闭合环）
        room_id: SurfaceId,
        /// 使用的公式（可选，默认 Sabine）
        formula: Option<ReverberationFormula>,
        /// 房间高度（可选，默认 3.0m）
        room_height: Option<f64>,
    },
    /// 多区域对比分析
    ComparativeAnalysis {
        /// 多个选区
        selections: Vec<NamedSelection>,
        /// 对比指标
        metrics: Vec<ComparisonMetric>,
    },
}

/// 选区边界
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "type", rename_all = "SCREAMING_SNAKE_CASE")]
pub enum SelectionBoundary {
    /// 矩形
    Rect { min: Point2, max: Point2 },
    /// 多边形
    Polygon { points: Vec<Point2> },
}

impl SelectionBoundary {
    /// 创建矩形边界
    pub fn rect(min: Point2, max: Point2) -> Self {
        Self::Rect { min, max }
    }

    /// 创建多边形边界
    pub fn polygon(points: Vec<Point2>) -> Self {
        Self::Polygon { points }
    }
}

/// 选区模式
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum SelectionMode {
    /// 完全包含
    Contained,
    /// 相交
    Intersecting,
    /// 智能（默认）
    #[default]
    Smart,
}

/// 命名选区（用于对比分析）
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct NamedSelection {
    /// 选区名称
    pub name: String,
    /// 选区边界
    pub boundary: SelectionBoundary,
}

/// 对比指标
#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ComparisonMetric {
    /// 面积
    Area,
    /// 平均吸声系数
    AverageAbsorption,
    /// 等效吸声面积
    EquivalentAbsorptionArea,
    /// 材料数量
    MaterialCount,
}

/// 混响时间公式
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ReverberationFormula {
    /// Sabine 公式：T60 = 0.161 × V / A
    /// 适用条件：α < 0.2（低吸声房间）
    #[default]
    Sabine,
    /// Eyring 公式：T60 = 0.161 × V / (-S × ln(1-α))
    /// 适用条件：α > 0.2（高吸声房间）
    Eyring,
    /// 自动选择：根据平均吸声系数 α 自动选择公式
    /// - α < 0.2: Sabine
    /// - α >= 0.2: Eyring
    Auto,
}

pub use common_types::acoustic::Frequency;

// ============================================================================
// 输出类型
// ============================================================================

/// 声学分析输出
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct AcousticOutput {
    /// 分析结果
    pub result: AcousticResult,
    /// 计算耗时
    pub computation_time: std::time::Duration,
    /// 指标
    pub metrics: AcousticMetrics,
}

/// 声学分析结果
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "type", rename_all = "SCREAMING_SNAKE_CASE")]
pub enum AcousticResult {
    /// 选区材料统计
    SelectionMaterialStats(SelectionMaterialStatsResult),
    /// 混响时间结果
    RoomReverberation(ReverberationResult),
    /// 对比分析结果
    ComparativeAnalysis(ComparativeAnalysisResult),
}

/// 选区材料统计结果
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct SelectionMaterialStatsResult {
    /// 选中的表面 ID
    pub surface_ids: Vec<SurfaceId>,
    /// 总表面积 (m²)
    pub total_area: f64,
    /// 材料分布
    pub material_distribution: Vec<MaterialDistribution>,
    /// 等效吸声面积 (频率相关，m²)
    pub equivalent_absorption_area: BTreeMap<Frequency, f64>,
    /// 平均吸声系数 (频率相关)
    pub average_absorption_coefficient: BTreeMap<Frequency, f64>,
}

/// 材料分布
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct MaterialDistribution {
    /// 材料名称
    pub material_name: String,
    /// 面积 (m²)
    pub area: f64,
    /// 百分比 (0-100)
    pub percentage: f64,
}

/// 混响时间结果
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ReverberationResult {
    /// 房间体积 (m³)
    pub volume: f64,
    /// 总表面积 (m²)
    pub total_surface_area: f64,
    /// 使用的公式
    pub formula: ReverberationFormula,
    /// T60 (频率相关，秒)
    pub t60: BTreeMap<Frequency, f64>,
    /// 早期衰变时间 EDT (频率相关，秒)
    pub edt: BTreeMap<Frequency, f64>,
}

/// 对比分析结果
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ComparativeAnalysisResult {
    /// 各区域统计
    pub regions: Vec<RegionStats>,
}

/// 区域统计
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct RegionStats {
    /// 区域名称
    pub name: String,
    /// 面积 (m²)
    pub area: f64,
    /// 材料数量
    pub material_count: usize,
    /// 平均吸声系数 (频率相关)
    pub average_absorption: BTreeMap<Frequency, f64>,
    /// 等效吸声面积 (频率相关，m²)
    pub equivalent_absorption_area: BTreeMap<Frequency, f64>,
}

/// 声学分析指标
#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
pub struct AcousticMetrics {
    /// 表面数量
    pub surface_count: usize,
    /// 计算耗时 (ms)
    pub computation_time_ms: f64,
}

// ============================================================================
// 错误类型
// ============================================================================

/// 声学分析错误恢复建议
#[derive(Debug, Clone)]
pub struct AcousticRecoverySuggestion {
    /// 人类可读的修复建议
    pub action: String,
    /// 建议的配置变更
    pub config_change: Option<(String, String)>,
    /// 优先级（1-10，10 为最高优先级）
    pub priority: u8,
}

impl AcousticRecoverySuggestion {
    /// 创建恢复建议
    pub fn new(action: impl Into<String>) -> Self {
        Self {
            action: action.into(),
            config_change: None,
            priority: 5,
        }
    }

    /// 设置配置变更建议
    pub fn with_config_change(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.config_change = Some((key.into(), value.into()));
        self
    }

    /// 设置优先级
    pub fn with_priority(mut self, priority: u8) -> Self {
        self.priority = priority.min(10);
        self
    }
}

impl std::fmt::Display for AcousticRecoverySuggestion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.action)?;
        if let Some((key, value)) = &self.config_change {
            write!(f, "（建议配置：{} = {}）", key, value)?;
        }
        Ok(())
    }
}

/// 声学分析错误
#[derive(Debug)]
pub enum AcousticError {
    /// 选区计算失败
    SelectionError {
        message: String,
        suggestion: Option<AcousticRecoverySuggestion>,
    },

    /// 房间体积计算失败
    VolumeCalculationFailed {
        message: String,
        suggestion: Option<AcousticRecoverySuggestion>,
    },

    /// 表面未分配材料
    MaterialNotAssigned {
        surface_id: SurfaceId,
        suggestion: Option<AcousticRecoverySuggestion>,
    },

    /// 无效的房间 ID
    InvalidRoomId {
        room_id: SurfaceId,
        suggestion: Option<AcousticRecoverySuggestion>,
    },

    /// 无效的选区（空选区）
    EmptySelection {
        suggestion: Option<AcousticRecoverySuggestion>,
    },

    /// 材料数据不完整
    IncompleteMaterialData {
        message: String,
        suggestion: Option<AcousticRecoverySuggestion>,
    },

    /// 计算失败（除零等）
    CalculationFailed {
        message: String,
        suggestion: Option<AcousticRecoverySuggestion>,
    },
}

impl std::fmt::Display for AcousticError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AcousticError::SelectionError {
                message,
                suggestion,
            } => {
                write!(f, "选区计算失败：{}", message)?;
                if let Some(s) = suggestion {
                    write!(f, "\n建议：{}", s)?;
                }
                Ok(())
            }
            AcousticError::VolumeCalculationFailed {
                message,
                suggestion,
            } => {
                write!(f, "房间体积计算失败：{}", message)?;
                if let Some(s) = suggestion {
                    write!(f, "\n建议：{}", s)?;
                }
                Ok(())
            }
            AcousticError::MaterialNotAssigned {
                surface_id,
                suggestion,
            } => {
                write!(f, "表面未分配材料：{:?}", surface_id)?;
                if let Some(s) = suggestion {
                    write!(f, "\n建议：{}", s)?;
                }
                Ok(())
            }
            AcousticError::InvalidRoomId {
                room_id,
                suggestion,
            } => {
                write!(f, "无效的房间 ID: {:?}", room_id)?;
                if let Some(s) = suggestion {
                    write!(f, "\n建议：{}", s)?;
                }
                Ok(())
            }
            AcousticError::EmptySelection { suggestion } => {
                write!(f, "无效的选区：未选中任何表面")?;
                if let Some(s) = suggestion {
                    write!(f, "\n建议：{}", s)?;
                }
                Ok(())
            }
            AcousticError::IncompleteMaterialData {
                message,
                suggestion,
            } => {
                write!(f, "材料数据不完整：{}", message)?;
                if let Some(s) = suggestion {
                    write!(f, "\n建议：{}", s)?;
                }
                Ok(())
            }
            AcousticError::CalculationFailed {
                message,
                suggestion,
            } => {
                write!(f, "计算失败：{}", message)?;
                if let Some(s) = suggestion {
                    write!(f, "\n建议：{}", s)?;
                }
                Ok(())
            }
        }
    }
}

impl std::error::Error for AcousticError {}

impl AcousticError {
    /// 创建选区错误
    pub fn selection(msg: impl Into<String>) -> Self {
        Self::SelectionError {
            message: msg.into(),
            suggestion: None,
        }
    }

    /// 创建选区错误（带建议）
    pub fn selection_with_suggestion(
        msg: impl Into<String>,
        suggestion: AcousticRecoverySuggestion,
    ) -> Self {
        Self::SelectionError {
            message: msg.into(),
            suggestion: Some(suggestion),
        }
    }

    /// 创建体积计算错误
    pub fn volume(msg: impl Into<String>) -> Self {
        Self::VolumeCalculationFailed {
            message: msg.into(),
            suggestion: None,
        }
    }

    /// 创建材料未分配错误
    pub fn material_not_assigned(id: SurfaceId) -> Self {
        Self::MaterialNotAssigned {
            surface_id: id,
            suggestion: Some(
                AcousticRecoverySuggestion::new("为该表面分配材料，或检查场景的材料配置")
                    .with_priority(7),
            ),
        }
    }

    /// 创建无效房间 ID 错误
    pub fn invalid_room_id(id: SurfaceId) -> Self {
        Self::InvalidRoomId {
            room_id: id,
            suggestion: Some(
                AcousticRecoverySuggestion::new(format!(
                    "房间 ID {} 不存在。请检查：\n1. 场景中是否有外轮廓（outer）或孔洞（holes）\n2. 房间 ID 范围：0 为外轮廓，1-N 为孔洞",
                    id
                ))
                .with_priority(8)
            ),
        }
    }

    /// 创建空选区错误
    pub fn empty_selection() -> Self {
        Self::EmptySelection {
            suggestion: Some(
                AcousticRecoverySuggestion::new(
                    "选区为空，请尝试：\n1. 扩大选区范围\n2. 检查选区坐标是否正确\n3. 确认场景中是否有边（当前边数：0）"
                )
                .with_priority(7)
            ),
        }
    }

    /// 创建材料数据不完整错误
    pub fn incomplete_material_data(msg: impl Into<String>) -> Self {
        Self::IncompleteMaterialData {
            message: msg.into(),
            suggestion: None,
        }
    }

    /// 创建计算失败错误
    pub fn calculation_failed(msg: impl Into<String>) -> Self {
        Self::CalculationFailed {
            message: msg.into(),
            suggestion: None,
        }
    }

    /// 获取恢复建议
    pub fn suggestion(&self) -> Option<&AcousticRecoverySuggestion> {
        match self {
            AcousticError::SelectionError { suggestion, .. } => suggestion.as_ref(),
            AcousticError::VolumeCalculationFailed { suggestion, .. } => suggestion.as_ref(),
            AcousticError::MaterialNotAssigned { suggestion, .. } => suggestion.as_ref(),
            AcousticError::InvalidRoomId { suggestion, .. } => suggestion.as_ref(),
            AcousticError::EmptySelection { suggestion } => suggestion.as_ref(),
            AcousticError::IncompleteMaterialData { suggestion, .. } => suggestion.as_ref(),
            AcousticError::CalculationFailed { suggestion, .. } => suggestion.as_ref(),
        }
    }
}

// ============================================================================
// 辅助函数
// ============================================================================

/// 计算等效吸声面积 A = Σ(S × α)
pub fn compute_equivalent_absorption_area(
    areas: &HashMap<String, f64>,
    absorption_coeffs: &HashMap<String, BTreeMap<Frequency, f64>>,
) -> BTreeMap<Frequency, f64> {
    let mut result: BTreeMap<Frequency, f64> = BTreeMap::new();

    for (material, &area) in areas {
        if let Some(coeffs) = absorption_coeffs.get(material) {
            for (freq, &coeff) in coeffs {
                *result.entry(*freq).or_insert(0.0) += area * coeff;
            }
        }
    }

    result
}

/// 计算平均吸声系数 α_avg = A / S_total
pub fn compute_average_absorption(
    equivalent_area: &BTreeMap<Frequency, f64>,
    total_area: f64,
) -> BTreeMap<Frequency, f64> {
    if total_area <= 0.0 {
        return BTreeMap::new();
    }

    equivalent_area
        .iter()
        .map(|(&freq, &area)| (freq, area / total_area))
        .collect()
}
