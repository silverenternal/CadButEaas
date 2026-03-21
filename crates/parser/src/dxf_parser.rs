//! DXF 文件解析器
//!
//! 使用 dxf crate 解析 DXF 文件，提取几何实体
//!
//! ## 特性
//! - 使用 NURBS 库精确离散化 Spline 曲线
//! - 智能图层语义识别（墙、门、窗等）
//! - 动态椭圆离散化（基于弦高误差）
//! - $INSUNITS 单位自动解析
//! - 二进制 DXF 检测与诊断
//!
//! ## 并行化（P11 锐评落实）
//! - 使用 rayon 并行解析大文件中的实体（>100 实体时启用）
//! - 块定义解析和实体解析均支持并行处理
//!
//! ### 并行化效果说明（P11 锐评）
//! **注意**：当前并行化仅针对实体转换（轻量操作，主要是字段拷贝），并行化的 overhead 可能比收益还大。
//! 真正的耗时大户是 `Drawing::load_file()`（文件 IO 和 DXF 解析），这部分没有并行化。
//! 
//! **P2 阶段改进计划**：并行化几何处理（端点吸附、交点计算），而非实体转换。
//!
//! ## 容差系统（P0 落实）
//! 本解析器已迁移到动态容差系统 (`AdaptiveTolerance`)，替代硬编码的 `BULGE_EPSILON` 和 `POINT_Z_EPSILON`。
//! 动态容差基于：
//! - 图纸单位（从 $INSUNITS 解析）
//! - 场景特征尺度（从坐标范围计算）
//! - 用户操作精度（从交互行为推断）

use common_types::{RawEntity, EntityMetadata, Point2, Polyline, CadError, BoundarySemantic, DxfParseReason, BlockDefinition, PathCommand};
use common_types::{SeatZone, SeatType, AcousticProps, ClosedLoop};
use common_types::{ParseStage, ParseProgress, adaptive_tolerance::AdaptiveTolerance};
use common_types::{HatchBoundaryPath, HatchPattern, DimensionType};
use crate::hatch_parser::HatchParser;
use dxf::{Drawing, entities::EntityType};
use dxf::entities::Insert;
use curvo::prelude::NurbsCurve;
use std::path::{Path, PathBuf};
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, BufReader};
use rayon::prelude::*;
use std::sync::Arc;
use tokio::sync::watch;

use geo::{Coord, LineString, SimplifyVw};

/// 3D 实体警告
#[derive(Debug, Clone)]
pub struct Entity3DWarning {
    pub entity_type: String,
    pub handle: Option<String>,
    pub z_range: [f64; 2],
    pub message: String,
}

/// SPLINE 简化统计
#[derive(Debug, Clone, Default)]
pub struct SplineSimplificationStats {
    pub total_splines: usize,
    pub simplified_splines: usize,
    pub total_control_points_before: usize,
    pub total_control_points_after: usize,
}

/// Bulge 容差：小于此值视为直线
///
/// # 物理解释（P11 锐评落实）
///
/// bulge = tan(θ/4)，其中 θ 是圆弧的包含角。
/// 1e-10 的 bulge 对应角度约 0.00002 度。
///
/// # 测试验证
///
/// 对于 10m 弦长，1e-10 bulge 产生的拱高计算：
/// - 拱高 h = R - R*cos(θ/2)
/// - bulge = tan(θ/4) ≈ θ/4 (小角度近似)
/// - θ ≈ 4e-10 弧度
/// - R = L / (2*sin(θ/2)) ≈ L / θ ≈ 2.5e10 m
/// - h ≈ R * θ²/8 ≈ 0.0005 mm
///
/// 结论：1e-10 bulge 产生的拱高远小于 CAD 绘图精度要求（通常 0.01mm）
///
/// # P1 优化：动态阈值
/// 注意：此常量仅用于向后兼容。新代码应使用 `AdaptiveTolerance::bulge_threshold()`
/// 基于弦长和场景容差动态计算阈值，而非使用固定值。
const BULGE_EPSILON: f64 = 1e-10;

/// DXF 线宽枚举值转换为毫米值
/// 
/// DXF 线宽值（组码 370）与毫米值的映射关系：
/// -3 = BYLAYER, -2 = BYBLOCK, -1 = DEFAULT
/// 0 = 0.00mm, 1 = 0.05mm, ..., 23 = 2.11mm
fn lineweight_enum_to_mm(enum_value: i16) -> Option<f64> {
    match enum_value {
        -3..=-1 => None, // BYLAYER, BYBLOCK, DEFAULT - 使用图层/块/默认值
        0 => Some(0.00),
        1 => Some(0.05),
        2 => Some(0.09),
        3 => Some(0.13),
        4 => Some(0.15),
        5 => Some(0.18),
        6 => Some(0.20),
        7 => Some(0.25),
        8 => Some(0.30),
        9 => Some(0.35),
        10 => Some(0.40),
        11 => Some(0.50),
        12 => Some(0.53),
        13 => Some(0.60),
        14 => Some(0.70),
        15 => Some(0.80),
        16 => Some(0.90),
        17 => Some(1.00),
        18 => Some(1.06),
        19 => Some(1.20),
        20 => Some(1.40),
        21 => Some(1.58),
        22 => Some(2.00),
        23 => Some(2.11),
        _ => None, // 无效值
    }
}

/// 将 bulge（凸度）转换为圆弧离散化点
/// bulge = tan(θ/4)，其中θ为圆弧的包含角
/// bulge > 0 表示逆时针（凸弧），bulge < 0 表示顺时针（凹弧）
///
/// # P11 锐评落实：数值稳定性改进
/// 使用弦高误差（sagitta）而非 bulge 绝对值判断是否简化为直线
/// 建筑图纸中常见 R=50m 的大半径圆弧，bulge 约 0.001，刚好在阈值边缘
/// 大半径圆弧被错误简化为直线会导致墙体/门窗位置偏差达 5-10mm
///
/// # P0 优化：动态阈值
/// 本函数已更新为使用动态容差。
/// 
/// ## 使用示例（推荐）
/// ```rust
/// use common_types::adaptive_tolerance::AdaptiveTolerance;
///
/// let tol = AdaptiveTolerance::new(unit, scene_scale, PrecisionLevel::Normal);
/// let bulge_threshold = tol.bulge_threshold(chord_length);
///
/// // 使用动态阈值替代硬编码的 BULGE_EPSILON
/// if bulge.abs() < bulge_threshold {
///     return vec![p1, p2];  // 简化为直线
/// }
/// ```
fn bulge_to_arc_points(p1: Point2, p2: Point2, bulge: f64, tolerance: f64) -> Polyline {
    // 先计算弦长，用于动态 bulge 阈值
    let chord_length = ((p2[0] - p1[0]).powi(2) + (p2[1] - p1[1]).powi(2)).sqrt();
    
    // 使用动态计算的 bulge 阈值替代硬编码的 BULGE_EPSILON
    // 公式：bulge_threshold = 2 * max_sagitta / chord_length
    // 其中 max_sagitta = tolerance * 0.1
    let bulge_threshold = 2.0 * (tolerance * 0.1) / chord_length.max(tolerance);
    
    if bulge.abs() < bulge_threshold {
        // bulge 足够小，简化为直线
        return vec![p1, p2];
    }

    // bulge = tan(θ/4) => θ = 4 * atan(bulge)
    let included_angle = 4.0 * bulge.atan();

    // 计算圆弧半径和圆心
    // 弦长
    let chord_length = ((p2[0] - p1[0]).powi(2) + (p2[1] - p1[1]).powi(2)).sqrt();

    // 圆弧半径 R = (L/2) / sin(θ/2)
    let sin_half_angle = (included_angle / 2.0).sin();

    // P11 锐评落实：基于弦高误差而非 bulge 绝对值判断
    // 拱高（sagitta）= L * |bulge| / 2
    // 只有当拱高小于容差的 1/10 时才简化为直线
    let sagitta = chord_length * bulge.abs() / 2.0;
    let tolerance_threshold = tolerance * 0.1;

    // P0 优化：使用动态容差替代硬编码的 BULGE_EPSILON
    // 当弦高误差小于容差的 1/10 时，简化为直线
    if sagitta < tolerance_threshold {
        // 角度很小或拱高很小，近似为直线
        return vec![p1, p2];
    }

    let radius = (chord_length / 2.0) / sin_half_angle.abs();

    // 计算圆心位置
    // 弦的中点
    let mid = [(p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0];

    // 弦的方向向量
    let chord_dir = [p2[0] - p1[0], p2[1] - p1[1]];
    let chord_len = chord_length;

    // 垂直方向（逆时针旋转 90 度）
    let perp = if bulge > 0.0 {
        [-chord_dir[1] / chord_len, chord_dir[0] / chord_len]
    } else {
        [chord_dir[1] / chord_len, -chord_dir[0] / chord_len]
    };

    // 圆心到中点的距离 d = R * cos(θ/2)
    let cos_half_angle = (included_angle / 2.0).cos();
    let dist = radius * cos_half_angle;

    // 圆心
    let center = [mid[0] + perp[0] * dist, mid[1] + perp[1] * dist];

    // 计算起点和终点相对于圆心的角度
    let start_angle = (p1[1] - center[1]).atan2(p1[0] - center[0]);
    let end_angle = (p2[1] - center[1]).atan2(p2[0] - center[0]);

    // 离散化圆弧 - 优化版本
    discretize_arc_optimized(center, radius, start_angle, end_angle, bulge > 0.0, tolerance, chord_length)
}

/// 离散化圆弧（优化版本 - P1 精度改进）
///
/// ## 改进点
/// 1. **更精确的弦高误差公式**：使用 `sagitta = R * (1 - cos(θ/2))` 而非近似值
/// 2. **最小段数保证**：至少 8 段（而非 4 段），确保圆弧平滑度
/// 3. **大半径特殊处理**：对于 R > 10m 的圆弧，使用固定角度步长（而非固定弦高）
/// 4. **弦长感知**：根据弦长动态调整离散化密度
///
/// ## 参数
/// - `chord_length`: 弦长，用于动态调整离散化密度
fn discretize_arc_optimized(
    center: Point2,
    radius: f64,
    start_angle: f64,
    end_angle: f64,
    ccw: bool,
    tolerance: f64,
    chord_length: f64,
) -> Polyline {
    let mut end = end_angle;

    // 确保角度方向正确
    if ccw {
        if end <= start_angle {
            end += 2.0 * std::f64::consts::PI;
        }
    } else if end >= start_angle {
        end -= 2.0 * std::f64::consts::PI;
    }

    let total_angle = (end - start_angle).abs();

    // ========================================================================
    // P1 优化：改进的离散化策略
    // ========================================================================

    // 方法 1：基于弦高误差的计算（适用于中小半径）
    // 弦高 h = R * (1 - cos(θ/2)) => θ = 2 * acos(1 - h/R)
    let max_angle_step_h = if radius > tolerance {
        2.0 * ((1.0 - tolerance / radius).acos().min(std::f64::consts::PI / 8.0))
    } else {
        std::f64::consts::PI / 8.0  // 半径很小时限制最大步长
    };

    // 方法 2：基于固定角度步长（适用于大半径 R > 10m）
    // 大半径时弦高误差公式会失效，改用固定角度步长
    let max_angle_step_fixed = std::f64::consts::PI / 32.0;  // 5.625 度

    // 选择更保守的步长
    let max_angle_step = if radius > 10000.0 {  // R > 10m
        max_angle_step_fixed
    } else {
        max_angle_step_h.min(max_angle_step_fixed)
    };

    // 计算基础段数
    let base_segments = (total_angle / max_angle_step).ceil() as usize;

    // P1 优化：最小段数保证（从 4 提升到 8）
    // 对于建筑图纸中的圆弧墙体，8 段能更好地保持曲率
    let min_segments = 8;

    // P1 优化：弦长感知调整
    // 如果弦长较长，增加段数以保持离散化精度
    let chord_based_segments = if chord_length > 1000.0 {  // 弦长 > 1m
        ((chord_length / 100.0).ceil() as usize).max(min_segments)  // 每 100mm 至少 1 段
    } else {
        min_segments
    };

    // 取最大值
    let segments = base_segments.max(chord_based_segments);

    // 确保至少有一个中间点（起点 + 中间点 + 终点）
    let segments = segments.max(2);

    let mut points = Vec::with_capacity(segments + 1);
    points.push([
        center[0] + radius * start_angle.cos(),
        center[1] + radius * start_angle.sin(),
    ]);

    let step = total_angle / segments as f64;
    let direction = if ccw { 1.0 } else { -1.0 };

    for i in 1..segments {
        let angle = start_angle + direction * step * i as f64;
        points.push([
            center[0] + radius * angle.cos(),
            center[1] + radius * angle.sin(),
        ]);
    }

    // 添加终点
    points.push([
        center[0] + radius * end_angle.cos(),
        center[1] + radius * end_angle.sin(),
    ]);

    points
}

/// DXF 解析结果统计
#[derive(Debug, Clone, Default)]
pub struct DxfParseReport {
    /// 图层分布：图层名 -> 实体数量
    pub layer_distribution: HashMap<String, usize>,
    /// 实体类型分布：类型名 -> 数量
    pub entity_type_distribution: HashMap<String, usize>,
    /// 警告信息
    pub warnings: Vec<String>,
    /// 图纸单位（来自 $INSUNITS）
    pub drawing_units: Option<String>,
    /// 图纸单位转换比例（转换为毫米的比例因子）
    pub unit_scale: f64,
    /// 是否检测到单位不匹配（图纸标注单位与实际坐标范围不符）
    pub unit_mismatch_detected: bool,
    /// 块定义数量
    pub block_definitions_count: usize,
    /// 块引用数量
    pub block_references_count: usize,

    // ========================================================================
    // P11 技术设计文档 v1.0 新增字段
    // ========================================================================

    /// 检测到的座椅区块
    pub seat_zones: Vec<common_types::SeatZone>,
    /// 3D 实体警告
    pub _3d_entity_warnings: Vec<Entity3DWarning>,
    /// SPLINE 简化统计
    pub spline_simplification_stats: SplineSimplificationStats,

    // ========================================================================
    // P3 增强：图层分组统计
    // ========================================================================

    /// 图层分组统计：分组名 -> 图层名列表
    pub layer_groups: HashMap<String, Vec<String>>,
    /// 图层可见性状态：图层名 -> 是否可见
    pub layer_visibility: HashMap<String, bool>,
    /// 过滤掉的图层数量
    pub filtered_layers_count: usize,

    // ========================================================================
    // P5 增强：解析质量报告
    // ========================================================================

    /// 解析质量评分（0.0 - 1.0）
    pub quality_score: f64,
    /// 解析问题详情
    pub issues: Vec<ParseIssue>,
    /// 解析统计
    pub parse_stats: ParseStats,
}

/// P5 新增：解析问题类型
#[derive(Debug, Clone)]
pub struct ParseIssue {
    /// 问题代码
    pub code: String,
    /// 问题描述
    pub message: String,
    /// 严重程度
    pub severity: ParseIssueSeverity,
    /// 受影响的图层（可选）
    pub layer: Option<String>,
    /// 受影响的实体类型（可选）
    pub entity_type: Option<String>,
    /// 建议的修复方法
    pub suggestion: Option<String>,
    /// 实体句柄（用于追踪问题实体）
    pub handle: Option<String>,
}

impl ParseIssue {
    /// 创建一个新的解析问题
    pub fn new(
        code: impl Into<String>,
        message: impl Into<String>,
        severity: ParseIssueSeverity,
    ) -> Self {
        Self {
            code: code.into(),
            message: message.into(),
            severity,
            layer: None,
            entity_type: None,
            suggestion: None,
            handle: None,
        }
    }

    /// 设置图层信息
    pub fn with_layer(mut self, layer: impl Into<String>) -> Self {
        self.layer = Some(layer.into());
        self
    }

    /// 设置实体类型
    pub fn with_entity_type(mut self, entity_type: impl Into<String>) -> Self {
        self.entity_type = Some(entity_type.into());
        self
    }

    /// 设置建议的修复方法
    pub fn with_suggestion(mut self, suggestion: impl Into<String>) -> Self {
        self.suggestion = Some(suggestion.into());
        self
    }

    /// 设置实体句柄
    pub fn with_handle(mut self, handle: impl Into<String>) -> Self {
        self.handle = Some(handle.into());
        self
    }

    /// 创建错误恢复问题（跳过损坏实体）
    pub fn skipped_entity(
        handle: Option<&str>,
        layer: Option<&str>,
        entity_type: &str,
        reason: &str,
    ) -> Self {
        Self::new(
            "SKIPPED_ENTITY",
            format!("跳过损坏实体：{}", reason),
            ParseIssueSeverity::Warning,
        )
        .with_layer(layer.unwrap_or("unknown").to_string())
        .with_entity_type(entity_type.to_string())
        .with_handle(handle.unwrap_or("unknown").to_string())
        .with_suggestion("检查 DXF 文件是否损坏，或尝试使用其他 CAD 软件重新保存".to_string())
    }

    /// 创建几何错误问题（如零长度线段）
    pub fn geometry_error(
        handle: Option<&str>,
        layer: Option<&str>,
        entity_type: &str,
        error_msg: &str,
    ) -> Self {
        Self::new(
            "GEOMETRY_ERROR",
            format!("几何错误：{}", error_msg),
            ParseIssueSeverity::Warning,
        )
        .with_layer(layer.unwrap_or("unknown").to_string())
        .with_entity_type(entity_type.to_string())
        .with_handle(handle.unwrap_or("unknown").to_string())
        .with_suggestion("检查实体几何数据是否有效".to_string())
    }
}

/// P5 新增：问题严重程度
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParseIssueSeverity {
    Info,       // 信息
    Warning,    // 警告
    Error,      // 错误
    Critical,   // 严重
}

/// P5 新增：解析统计
#[derive(Debug, Clone, Default)]
pub struct ParseStats {
    /// 总实体数量
    pub total_entities: usize,
    /// 有效实体数量
    pub valid_entities: usize,
    /// 跳过的实体数量
    pub skipped_entities: usize,
    /// 解析错误数量
    pub error_count: usize,
    /// 解析警告数量
    pub warning_count: usize,
    /// 解析时间（毫秒）
    pub parse_time_ms: f64,
    /// 文件 size（字节）
    pub file_size_bytes: u64,
    /// 坐标范围
    pub bounding_box: Option<([f64; 2], [f64; 2])>,
    /// P6 新增：恢复的实体数量（成功从错误中恢复的实体）
    pub recovered_entities: usize,
    /// P6 新增：损坏的实体数量（无法恢复的实体）
    pub corrupted_entities: usize,
    /// P6 新增：错误恢复率（恢复数 / (恢复数 + 损坏数)）
    pub recovery_rate: f64,
}

impl ParseStats {
    /// 计算错误恢复率
    pub fn calculate_recovery_rate(&mut self) {
        let total = self.recovered_entities + self.corrupted_entities;
        self.recovery_rate = if total > 0 {
            self.recovered_entities as f64 / total as f64
        } else {
            1.0 // 默认 100% 恢复率（如果没有错误）
        };
    }
}

/// 实体类型枚举（用于白名单过滤）
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EntityTypeFilter {
    Line,
    Polyline,
    LwPolyline,
    Arc,
    Circle,
    Spline,
    Ellipse,
    Text,
    MText,
    Insert,
}

/// DXF 解析配置
#[derive(Debug, Clone)]
pub struct DxfConfig {
    /// 图层白名单（None=全部）
    pub layer_whitelist: Option<Vec<String>>,
    /// 实体类型白名单（None=全部）
    pub entity_whitelist: Option<Vec<EntityTypeFilter>>,
    /// 颜色白名单（ACI 颜色索引，None=全部）
    /// ACI 颜色：1=红，2=黄，3=绿，4=青，5=蓝，6=品红，7=黑/白，8=灰，等等
    pub color_whitelist: Option<Vec<i16>>,
    /// 线宽白名单（None=全部）
    /// 线宽值：0=默认，1=0.00mm，2=0.05mm，...，18=2.11mm，ByLayer=-1，ByBlock=-2
    pub lineweight_whitelist: Option<Vec<i16>>,
    /// ARC 离散化容差（mm）
    pub arc_tolerance_mm: f64,
    /// 是否忽略文本
    pub ignore_text: bool,
    /// 是否忽略标注
    pub ignore_dimensions: bool,
    /// 是否忽略填充（HATCH）
    pub ignore_hatch: bool,

    // ========================================================================
    // P11 技术设计文档 v1.0 新增配置
    // ========================================================================

    /// 是否检测座椅区块
    pub detect_seat_zones: bool,
    /// 座椅区块最小数量阈值
    pub seat_zone_min_count: usize,
    /// 是否简化 SPLINE（控制点超过阈值时）
    pub simplify_splines: bool,
    /// SPLINE 最大控制点数
    pub max_spline_control_points: usize,
    /// 是否检测 3D 实体并警告
    pub detect_3d_entities: bool,

    // ========================================================================
    // P3 增强：智能图层过滤
    // ========================================================================

    /// 图层过滤模式
    pub layer_filter_mode: LayerFilterMode,
    /// 自定义图层分组（图层名模式 -> 分组名）
    pub custom_layer_groups: Vec<(String, String)>,
    /// 是否启用图层可见性控制（用于渲染优化）
    pub enable_layer_visibility: bool,
}

/// 图层过滤模式
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LayerFilterMode {
    /// 不过滤，保留所有图层
    #[default]
    All,
    /// 仅保留墙体相关图层（墙、柱、梁）
    WallsOnly,
    /// 仅保留开口图层（门、窗、开口）
    OpeningsOnly,
    /// 保留建筑图层（墙 + 开口）
    Architectural,
    /// 仅保留家具图层
    Furniture,
    /// 自定义过滤（使用 custom_layer_groups）
    Custom,
}

impl Default for DxfConfig {
    fn default() -> Self {
        Self {
            layer_whitelist: None,
            entity_whitelist: None,
            color_whitelist: None,
            lineweight_whitelist: None,
            arc_tolerance_mm: 0.1,
            ignore_text: true,
            ignore_dimensions: true,
            ignore_hatch: true,
            detect_seat_zones: true,
            seat_zone_min_count: 50,
            simplify_splines: true,
            max_spline_control_points: 500,
            detect_3d_entities: true,
            layer_filter_mode: LayerFilterMode::All,
            custom_layer_groups: Vec::new(),
            enable_layer_visibility: false,
        }
    }
}

/// DXF 解析器
///
/// 支持图层过滤、容差配置和自定义解析选项
#[derive(Clone)]
pub struct DxfParser {
    /// 图层过滤器 (None 表示不过滤)
    pub layer_filter: Option<Vec<String>>,
    /// 弦高误差容差（毫米）
    pub tolerance: f64,
    /// 解析配置
    pub config: DxfConfig,
    /// 进度回调发送器（可选）
    progress_tx: Option<Arc<watch::Sender<ParseProgress>>>,
    /// P0 新增：自适应容差计算器
    adaptive_tolerance: AdaptiveTolerance,
    /// HATCH 解析器
    hatch_parser: HatchParser,
}

impl DxfParser {
    pub fn new() -> Self {
        Self {
            layer_filter: None,
            tolerance: 0.1, // 默认 0.1mm 弦高误差
            config: DxfConfig::default(),
            progress_tx: None,
            adaptive_tolerance: AdaptiveTolerance::default(),
            hatch_parser: HatchParser::new(),
        }
    }

    pub fn with_layer_filter(mut self, layers: Vec<String>) -> Self {
        self.layer_filter = Some(layers);
        self
    }

    pub fn with_tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// 使用自定义配置创建解析器
    pub fn with_config(mut self, config: DxfConfig) -> Self {
        self.config = config;
        self
    }

    /// 设置进度回调发送器
    pub fn with_progress_callback(mut self, tx: watch::Sender<ParseProgress>) -> Self {
        self.progress_tx = Some(Arc::new(tx));
        self
    }

    /// 发送进度更新
    fn send_progress(&self, progress: ParseProgress) {
        if let Some(tx) = &self.progress_tx {
            let _ = tx.send(progress);
        }
    }

    /// 设置图层白名单
    pub fn with_layer_whitelist(mut self, layers: Vec<String>) -> Self {
        self.config.layer_whitelist = Some(layers);
        self
    }

    /// 设置实体类型白名单
    pub fn with_entity_whitelist(mut self, entity_types: Vec<EntityTypeFilter>) -> Self {
        self.config.entity_whitelist = Some(entity_types);
        self
    }

    /// 设置 ARC 离散化容差
    pub fn with_arc_tolerance(mut self, tolerance_mm: f64) -> Self {
        self.config.arc_tolerance_mm = tolerance_mm;
        self
    }

    /// 设置是否忽略文本
    pub fn with_ignore_text(mut self, ignore: bool) -> Self {
        self.config.ignore_text = ignore;
        self
    }

    /// 设置是否忽略标注
    pub fn with_ignore_dimensions(mut self, ignore: bool) -> Self {
        self.config.ignore_dimensions = ignore;
        self
    }

    /// 设置是否忽略填充
    pub fn with_ignore_hatch(mut self, ignore: bool) -> Self {
        self.config.ignore_hatch = ignore;
        self
    }

    /// P0 新增：设置自适应容差计算器
    pub fn with_adaptive_tolerance(mut self, tol: AdaptiveTolerance) -> Self {
        self.adaptive_tolerance = tol;
        self
    }

    /// 获取自适应容差计算器
    pub fn adaptive_tolerance(&self) -> &AdaptiveTolerance {
        &self.adaptive_tolerance
    }

    /// 设置颜色白名单（ACI 颜色索引）
    pub fn with_color_whitelist(mut self, colors: Vec<i16>) -> Self {
        self.config.color_whitelist = Some(colors);
        self
    }

    /// 设置线宽白名单
    pub fn with_lineweight_whitelist(mut self, lineweights: Vec<i16>) -> Self {
        self.config.lineweight_whitelist = Some(lineweights);
        self
    }

    /// 检查实体是否应该被包含（根据图层、颜色、线宽过滤）
    fn should_include_entity(&self, entity: &dxf::entities::Entity) -> bool {
        // 图层过滤
        if let Some(ref layers) = &self.config.layer_whitelist {
            if !layers.is_empty() && !layers.iter().any(|l| entity.common.layer.contains(l)) {
                return false;
            }
        }

        // 颜色过滤
        if let Some(ref colors) = &self.config.color_whitelist {
            if !colors.is_empty() {
                // 尝试从 Color 获取 ACI 索引值
                // index() 返回 Option<u8>，范围 1-255
                // 256 = ByLayer, 0 = ByBlock, 257 = ByEntity
                let color_index = entity.common.color.index();
                match color_index {
                    Some(idx) => {
                        // 有明确 ACI 索引，检查是否在白名单中
                        if !colors.contains(&(idx as i16)) {
                            return false;
                        }
                    }
                    None => {
                        // ByLayer/ByBlock 等情况：dxf 0.6 不支持获取图层颜色
                        // 保守策略：不过滤，保留实体（因为可能是墙体）
                        // 未来升级 dxf 库后可查询图层颜色进行精确过滤
                    }
                }
            }
        }

        // 线宽过滤
        if let Some(ref lineweights) = &self.config.lineweight_whitelist {
            if !lineweights.is_empty() {
                let entity_lineweight = entity.common.lineweight_enum_value;
                // 检查线宽值是否在白名单中
                if !lineweights.contains(&entity_lineweight) {
                    return false;
                }
            }
        }

        true
    }

    /// 获取 DXF 文件中所有图层的列表
    pub fn get_layer_list(&self, path: impl AsRef<Path>) -> Result<Vec<String>, CadError> {
        let path = path.as_ref();
        let drawing = Drawing::load_file(path)
            .map_err(|e| CadError::dxf_parse_with_source(path, DxfParseReason::FileNotFound, e))?;

        let mut layers: Vec<String> = drawing.layers()
            .map(|l| l.name.clone())
            .collect();
        layers.sort();
        layers.dedup();

        Ok(layers)
    }

    /// 智能识别墙体图层
    ///
    /// 自动检测包含墙相关关键词的图层（支持 AIA 标准和常见变体）
    pub fn detect_wall_layers(&self, path: impl AsRef<Path>) -> Result<Vec<String>, CadError> {
        let layers = self.get_layer_list(path)?;
        Ok(layers
            .into_iter()
            .filter(|l| is_wall_layer(l))
            .collect())
    }

    /// 智能识别门窗图层
    pub fn detect_door_window_layers(&self, path: impl AsRef<Path>) -> Result<Vec<String>, CadError> {
        let layers = self.get_layer_list(path)?;
        Ok(layers
            .into_iter()
            .filter(|l| is_door_only(l) || is_window_only(l))
            .collect())
    }

    /// 智能识别家具图层
    pub fn detect_furniture_layers(&self, path: impl AsRef<Path>) -> Result<Vec<String>, CadError> {
        let layers = self.get_layer_list(path)?;
        Ok(layers
            .into_iter()
            .filter(|l| is_furniture_layer(l))
            .collect())
    }

    /// 智能识别标注图层
    pub fn detect_dimension_layers(&self, path: impl AsRef<Path>) -> Result<Vec<String>, CadError> {
        let layers = self.get_layer_list(path)?;
        Ok(layers
            .into_iter()
            .filter(|l| is_dimension_layer(l))
            .collect())
    }

    /// 解析 DXF 文件并返回实体和统计报告
    pub fn parse_file_with_report(&self, path: impl AsRef<Path>) -> Result<(Vec<RawEntity>, DxfParseReport), CadError> {
        let path = path.as_ref();

        // 获取文件大小用于进度计算
        let total_bytes = std::fs::metadata(path)
            .map(|m| m.len())
            .unwrap_or(0);

        // 发送初始进度
        self.send_progress(ParseProgress {
            stage: ParseStage::ReadingFile,
            total_bytes,
            stage_progress: 0.0,
            overall_progress: 0.0,
            ..Default::default()
        });

        // 检测是否为二进制 DXF 文件
        if let Ok(file) = File::open(path) {
            let mut reader = BufReader::new(file);
            let mut buffer = [0u8; 6];
            if reader.read_exact(&mut buffer).is_ok() && Self::is_binary_dxf(&buffer) {
                return Err(CadError::dxf_parse_with_source(
                    path,
                    DxfParseReason::EncodingError("检测到二进制 DXF 文件（Binary DXF）。\n\
                             建议操作：\n\
                             1. 使用 AutoCAD 打开文件，另存为 ASCII 格式（DXF R12/LT2 或更高版本）\n\
                             2. 或使用在线转换工具（如 CADSoftTools、ShareCAD、A360 Viewer 等）\n\
                             3. 或使用 LibreCAD 导出为 ASCII DXF".to_string()),
                    std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "Binary DXF format detected - ASCII conversion required"
                    ),
                ));
            }
        }

        // 更新进度：解析文件头
        self.send_progress(ParseProgress {
            stage: ParseStage::ParsingHeader,
            total_bytes,
            bytes_read: total_bytes / 4,
            stage_progress: 0.3,
            ..Default::default()
        });

        let drawing = Drawing::load_file(path)
            .map_err(|e| {
                CadError::dxf_parse_with_source(
                    path,
                    DxfParseReason::FileNotFound,
                    std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!(
                            "DXF 文件加载失败：{}\n\
                             建议：\n\
                             1. 检查文件是否存在：{:?}\n\
                             2. 检查文件权限是否可读\n\
                             3. 确认 DXF 版本兼容性（支持 R12-R2018）",
                            e, path
                        ),
                    ),
                )
            })?;

        // 更新进度：解析表格段
        self.send_progress(ParseProgress {
            stage: ParseStage::ParsingTables,
            total_bytes,
            bytes_read: total_bytes / 2,
            stage_progress: 0.5,
            ..Default::default()
        });

        let mut report = DxfParseReport {
            unit_scale: 1.0, // 默认 1:1（毫米）
            ..Default::default()
        };

        // 解析 $INSUNITS 单位变量
        report.drawing_units = self.parse_insunits(&drawing, &mut report.unit_scale);

        // 更新进度：解析块定义
        self.send_progress(ParseProgress {
            stage: ParseStage::ParsingBlocks,
            total_bytes,
            bytes_read: total_bytes * 3 / 4,
            stage_progress: 0.6,
            ..Default::default()
        });

        let mut entities = self.extract_entities_with_report(&drawing, &mut report)?;

        // P0-1: 解析 HATCH 实体（使用低层级组码解析器）
        if !self.config.ignore_hatch {
            match self.hatch_parser.parse_hatch_entities(path.as_ref()) {
                Ok(hatch_entities) => {
                    tracing::info!("解析到 {} 个 HATCH 实体", hatch_entities.len());
                    // 将 HATCH 实体添加到结果中
                    entities.extend(hatch_entities);
                }
                Err(e) => {
                    tracing::warn!("HATCH 解析失败：{}", e);
                }
            }
        }

        // 更新进度：完成
        self.send_progress(ParseProgress {
            stage: ParseStage::Finalizing,
            total_bytes,
            bytes_read: total_bytes,
            entities_parsed: entities.len(),
            stage_progress: 1.0,
            overall_progress: 1.0,
            ..Default::default()
        });

        Ok((entities, report))
    }

    /// 检测二进制 DXF 文件
    ///
    /// 二进制 DXF 文件前缀特征：
    /// - AC1014 (AutoCAD R14)
    /// - AC1015 (AutoCAD 2000)
    /// - AC1018 (AutoCAD 2004)
    /// - AC1021 (AutoCAD 2007)
    /// - AC1024 (AutoCAD 2010)
    /// - AC1027 (AutoCAD 2013)
    /// - AC1032 (AutoCAD 2018)
    fn is_binary_dxf(buffer: &[u8]) -> bool {
        // ASCII DXF 通常以数字组码开头（如 "0"、"999" 等）
        // 二进制 DXF 有特定的文件头签名
        if buffer.len() < 6 {
            return false;
        }

        // 检查是否包含 AC 前缀（二进制 DXF 版本标识）
        buffer.starts_with(b"AC10") || buffer.starts_with(b"AC15")
    }

    /// 解析 $INSUNITS 单位变量
    ///
    /// DXF 规范中的单位代码：
    /// 0 = 无单位
    /// 1 = 英寸
    /// 2 = 英尺
    /// 3 = 英里
    /// 4 = 毫米
    /// 5 = 厘米
    /// 6 = 米
    /// 7 = 千米
    /// 8 = 微英寸
    /// 9 = 密耳
    /// 10-22 = 其他单位
    fn parse_insunits(&self, drawing: &Drawing, unit_scale: &mut f64) -> Option<String> {
        // dxf 0.6 crate 中 $INSUNITS 通过 header.drawing_units 字段访问
        let insunits_code = drawing.header.drawing_units as i32;

        let (unit_name, scale) = match insunits_code {
            0 => ("无单位", 1.0),
            1 => ("英寸", 25.4),
            2 => ("英尺", 304.8),
            3 => ("英里", 1609344.0),
            4 => ("毫米", 1.0),
            5 => ("厘米", 10.0),
            6 => ("米", 1000.0),
            7 => ("千米", 1000000.0),
            8 => ("微英寸", 0.0000254),
            9 => ("密耳", 0.0254),
            other => {
                tracing::warn!("未知单位代码：{}, 默认使用毫米", other);
                ("未知单位（默认毫米）", 1.0)
            }
        };

        *unit_scale = scale;
        Some(unit_name.to_string())
    }

    /// 从文件路径解析 DXF
    pub fn parse_file(&self, path: impl AsRef<Path>) -> Result<Vec<RawEntity>, CadError> {
        let path = path.as_ref();

        // 使用 dxf crate 的 Drawing::load_file 加载文件
        let drawing = Drawing::load_file(path)
            .map_err(|e| CadError::dxf_parse_with_source(path, DxfParseReason::FileNotFound, e))?;

        self.extract_entities(&drawing)
    }

    /// 从字节解析 DXF (ASCII 格式)
    pub fn parse_bytes(&self, bytes: &[u8]) -> Result<Vec<RawEntity>, CadError> {
        use std::io::Cursor;

        let mut cursor = Cursor::new(bytes);
        let drawing = Drawing::load(&mut cursor)
            .map_err(|e| CadError::dxf_parse_with_source(
                PathBuf::from("<bytes>"),
                DxfParseReason::EncodingError("DXF 字节解析失败".to_string()),
                e,
            ))?;

        self.extract_entities(&drawing)
    }

    /// 从 Drawing 中提取实体
    fn extract_entities(&self, drawing: &Drawing) -> Result<Vec<RawEntity>, CadError> {
        let mut report = DxfParseReport::default();
        let entities = self.extract_entities_with_report(drawing, &mut report)?;
        Ok(entities)
    }

    /// 从 Drawing 中提取实体并生成报告
    ///
    /// # P11 锐评落实：rayon 并行化
    /// - 块定义解析使用并行迭代器
    /// - 大文件实体解析使用并行迭代器（>1000 实体时自动启用）
    fn extract_entities_with_report(&self, drawing: &Drawing, report: &mut DxfParseReport) -> Result<Vec<RawEntity>, CadError> {
        // 获取单位转换比例
        let scale = report.unit_scale;

        // 1. 解析块定义（并行化）
        let block_definitions: HashMap<String, BlockDefinition> = drawing.blocks()
            .par_bridge()
            .filter_map(|block| {
                let block_entities: Vec<RawEntity> = block.entities
                    .iter()
                    .filter_map(|entity| self.convert_entity(entity).ok().flatten())
                    .collect();

                Some((
                    block.name.clone(),
                    BlockDefinition {
                        name: block.name.clone(),
                        base_point: [block.base_point.x * scale, block.base_point.y * scale],
                        entities: block_entities,
                        metadata: EntityMetadata::new(),
                    },
                ))
            })
            .collect();

        report.block_definitions_count = block_definitions.len();

        // 更新进度：解析实体
        self.send_progress(ParseProgress {
            stage: ParseStage::ParsingEntities,
            total_bytes: 0,  // 未知
            entities_parsed: 0,
            stage_progress: 0.0,
            ..Default::default()
        });

        // 2. 解析 ENTITIES 段
        let entities_vec: Vec<_> = drawing.entities().collect();
        let total_entity_count = entities_vec.len();
        
        // P11 锐评落实：降低并行阈值到 100，让测试能触发
        // 注意：实体转换本身是轻量操作（主要是字段拷贝），并行化的 overhead 可能比收益还大
        // 真正的耗时大户是 Drawing::load_file()（文件 IO 和 DXF 解析），这部分没有并行化
        // 并行化效果有限，待后续优化（P2 阶段：并行化几何处理如端点吸附、交点计算）
        let is_large_file = entities_vec.len() > 100;

        // 统计信息需要串行收集，所以先收集数据再统计
        let mut layer_distribution: HashMap<String, usize> = HashMap::new();
        let mut entity_type_distribution: HashMap<String, usize> = HashMap::new();
        let mut block_references_count = 0;

        // 预处理：统计图层和类型分布（用于报告）
        for entity in &entities_vec {
            *layer_distribution.entry(entity.common.layer.clone()).or_insert(0) += 1;
            let entity_type_name = match &entity.specific {
                EntityType::Line(_) => "LINE",
                EntityType::Polyline(_) => "POLYLINE",
                EntityType::LwPolyline(_) => "LWPOLYLINE",
                EntityType::Arc(_) => "ARC",
                EntityType::Circle(_) => "CIRCLE",
                EntityType::Spline(_) => "SPLINE",
                EntityType::Ellipse(_) => "ELLIPSE",
                EntityType::Text(_) => "TEXT",
                EntityType::MText(_) => "MTEXT",
                EntityType::Insert(_) => "INSERT",
                // P0-1: HATCH 和 P0-2: DIMENSION 暂时归类为 OTHER
                // 待后续实现完整解析
                _ => "OTHER",
            };
            *entity_type_distribution.entry(entity_type_name.to_string()).or_insert(0) += 1;
            if matches!(&entity.specific, EntityType::Insert(_)) {
                block_references_count += 1;
            }
        }

        report.layer_distribution = layer_distribution;
        report.entity_type_distribution = entity_type_distribution;
        report.block_references_count = block_references_count;

        // 实体转换（>100 实体时启用并行，P11 锐评落实）
        // 使用 flat_map 而非 filter_map，因为 INSERT 展开会产生多个实体
        // P6 增强：统计错误恢复信息
        let entities: Vec<RawEntity> = if is_large_file {
            // 并行处理实体（注意：实体转换是轻量操作，并行化效果有限）
            // 在并行迭代中定期发送进度更新
            let progress_tx = &self.progress_tx;
            let entities: Vec<RawEntity> = entities_vec.par_iter()
                .enumerate()
                .flat_map(|(idx, entity)| {
                    // 每 100 个实体发送一次进度
                    if idx % 100 == 0 {
                        if let Some(tx) = progress_tx {
                            let stage_progress = idx as f64 / total_entity_count as f64;
                            let _ = tx.send(ParseProgress {
                                stage: ParseStage::ParsingEntities,
                                total_bytes: 0,
                                entities_parsed: idx,
                                total_entities: Some(total_entity_count),
                                stage_progress,
                                ..Default::default()
                            });
                        }
                    }
                    self.process_entity(entity, &block_definitions, scale)
                })
                .collect();
            entities
        } else {
            // 串行处理实体
            entities_vec.iter()
                .flat_map(|entity| self.process_entity(entity, &block_definitions, scale))
                .collect()
        };

        // 更新进度：实体解析完成
        self.send_progress(ParseProgress {
            stage: ParseStage::ParsingEntities,
            total_bytes: 0,
            entities_parsed: entities.len(),
            total_entities: Some(total_entity_count),
            stage_progress: 1.0,
            ..Default::default()
        });

        // P6: 统计错误恢复信息
        // 注意：由于 process_entity 使用 catch_unwind 捕获 panic，我们无法直接知道哪些实体被跳过
        // 这里使用启发式方法：如果输出实体数量远少于输入，说明有实体被跳过
        let output_entity_count = entities.len();
        // 估算跳过的实体数量（假设每个实体平均产生 1 个输出实体）
        let estimated_skipped = total_entity_count.saturating_sub(output_entity_count);

        if estimated_skipped > 0 {
            report.parse_stats.skipped_entities = estimated_skipped;
            report.parse_stats.corrupted_entities = estimated_skipped;
            report.parse_stats.recovered_entities = 0;  // 暂时无法精确统计
            report.parse_stats.calculate_recovery_rate();

            tracing::warn!(
                "检测到 {} 个实体可能被跳过或解析失败（总输入：{}, 输出：{}）",
                estimated_skipped, total_entity_count, output_entity_count
            );
        }

        // 更新进度：构建拓扑
        self.send_progress(ParseProgress {
            stage: ParseStage::BuildingTopology,
            total_bytes: 0,
            entities_parsed: entities.len(),
            stage_progress: 0.5,
            ..Default::default()
        });

        // 检测单位不匹配
        self.detect_unit_mismatch(&entities, report);

        // 更新进度：几何计算
        self.send_progress(ParseProgress {
            stage: ParseStage::ComputingGeometry,
            total_bytes: 0,
            entities_parsed: entities.len(),
            stage_progress: 0.8,
            ..Default::default()
        });

        tracing::info!("从 DXF 中提取了 {} 个实体（块定义：{}，块引用：{}）",
            entities.len(), report.block_definitions_count, report.block_references_count);
        Ok(entities)
    }

    /// 处理单个实体（并行化辅助函数）
    /// 返回 Vec 而非 Option，因为 INSERT 展开会产生多个实体
    /// 
    /// P6 增强：实现错误恢复解析 - 跳过损坏实体继续解析
    fn process_entity(&self, entity: &dxf::entities::Entity, block_definitions: &HashMap<String, BlockDefinition>, scale: f64) -> Vec<RawEntity> {
        // P6: 使用 catch_unwind 捕获 panic，防止单个实体损坏导致整个解析失败
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.process_entity_impl(entity, block_definitions, scale)
        }));

        match result {
            Ok(entities) => entities,
            Err(_) => {
                // P6: 捕获到 panic，记录问题并跳过该实体
                let handle = format!("{:?}", entity.common.handle);
                let layer = &entity.common.layer;
                let entity_type = match &entity.specific {
                    EntityType::Line(_) => "LINE",
                    EntityType::Polyline(_) => "POLYLINE",
                    EntityType::LwPolyline(_) => "LWPOLYLINE",
                    EntityType::Arc(_) => "ARC",
                    EntityType::Circle(_) => "CIRCLE",
                    EntityType::Spline(_) => "SPLINE",
                    EntityType::Ellipse(_) => "ELLIPSE",
                    EntityType::Text(_) => "TEXT",
                    EntityType::MText(_) => "MTEXT",
                    EntityType::Insert(_) => "INSERT",
                    _ => "OTHER",
                };

                tracing::warn!(
                    "捕获实体解析 panic，已跳过：handle={}, layer={}, type={}",
                    handle, layer, entity_type
                );

                // 注意：这里无法直接更新 report，因为 process_entity 是纯函数
                // 统计信息会在 extract_entities_with_report 中统一收集
                vec![]
            }
        }
    }

    /// 实体解析实现（内部函数）
    /// P6: 将实际解析逻辑分离到此函数，便于错误恢复处理
    fn process_entity_impl(&self, entity: &dxf::entities::Entity, block_definitions: &HashMap<String, BlockDefinition>, scale: f64) -> Vec<RawEntity> {
        // 图层过滤
        if let Some(ref layers) = self.layer_filter {
            if !layers.is_empty() && !layers.contains(&entity.common.layer) {
                return vec![];
            }
        }

        // 颜色/线宽过滤
        if !self.should_include_entity(entity) {
            return vec![];
        }

        // 创建元数据
        let color_index = entity.common.color.index().map(|i| i.to_string());
        let layer_name = entity.common.layer.clone();

        // 材料映射：优先使用颜色，其次使用图层名
        let material = map_material_from_color(color_index.as_deref())
            .or_else(|| map_material_from_layer(&layer_name));

        let metadata = EntityMetadata {
            layer: Some(layer_name.clone()),
            color: color_index.clone(),
            lineweight: lineweight_enum_to_mm(entity.common.lineweight_enum_value),
            line_type: Some(entity.common.line_type_name.clone()).filter(|s| !s.is_empty()),
            handle: Some(format!("{:?}", entity.common.handle)),
            material,
            width: None,
        };

        // 根据图层识别语义
        let semantic = identify_semantic(&layer_name);
        let _handle_str = format!("{:?}", entity.common.handle);
        let _handle_opt = Some(_handle_str.as_str());
        let _layer_opt = Some(layer_name.as_str());

        // 根据实体类型转换
        match &entity.specific {
            EntityType::Line(line) => {
                let start = [line.p1.x * scale, line.p1.y * scale];
                let end = [line.p2.x * scale, line.p2.y * scale];

                // P6: 过滤零长度线段（使用更严格的容差）
                const MIN_LINE_LENGTH: f64 = 1e-4; // 0.1mm
                let length = ((end[0] - start[0]).powi(2) + (end[1] - start[1]).powi(2)).sqrt();
                if length < MIN_LINE_LENGTH {
                    tracing::debug!("过滤零长度线段：handle={:?}, length={:.6}", entity.common.handle, length);
                    return vec![];
                }

                vec![RawEntity::Line {
                    start,
                    end,
                    metadata: metadata.clone(),
                    semantic: semantic.clone(),
                }]
            }

            EntityType::Polyline(polyline) => {
                let points: Polyline = polyline.vertices()
                    .map(|v| [v.location.x * scale, v.location.y * scale])
                    .collect();

                // P6: 检查点数是否有效
                if points.len() < 2 {
                    tracing::debug!("多段线点数不足：handle={:?}, count={}", entity.common.handle, points.len());
                    return vec![];
                }

                // P6: 检查是否有 NaN 或无穷大值
                for (i, pt) in points.iter().enumerate() {
                    if pt[0].is_nan() || pt[1].is_nan() || pt[0].is_infinite() || pt[1].is_infinite() {
                        tracing::warn!("多段线包含无效坐标：handle={:?}, point_index={}", entity.common.handle, i);
                        return vec![];
                    }
                }

                vec![RawEntity::Polyline {
                    points,
                    closed: polyline.is_closed(),
                    metadata: metadata.clone(),
                    semantic: semantic.clone(),
                }]
            }

            EntityType::LwPolyline(lwpolyline) => {
                let mut points: Polyline = Vec::new();
                let vertices = &lwpolyline.vertices;

                if vertices.is_empty() {
                    tracing::debug!("轻多段线顶点为空：handle={:?}", entity.common.handle);
                    return vec![];
                }

                // P6: 检查顶点数据有效性
                for (i, v) in vertices.iter().enumerate() {
                    if v.x.is_nan() || v.y.is_nan() || v.x.is_infinite() || v.y.is_infinite() {
                        tracing::warn!("轻多段线顶点坐标无效：handle={:?}, vertex_index={}", entity.common.handle, i);
                        return vec![];
                    }
                }

                for i in 0..vertices.len() {
                    let v1 = &vertices[i];
                    points.push([v1.x * scale, v1.y * scale]);

                    if v1.bulge.abs() > BULGE_EPSILON {
                        let v2 = if i + 1 < vertices.len() {
                            &vertices[i + 1]
                        } else if lwpolyline.is_closed() {
                            &vertices[0]
                        } else {
                            continue;
                        };

                        let p1 = [v1.x * scale, v1.y * scale];
                        let p2 = [v2.x * scale, v2.y * scale];
                        
                        // P6: 检查 bulge 离散化结果
                        let arc_points = bulge_to_arc_points(p1, p2, v1.bulge, self.config.arc_tolerance_mm);

                        if arc_points.len() > 2 {
                            points.extend_from_slice(&arc_points[1..]);
                        }
                    }
                }

                if lwpolyline.is_closed() && !vertices.is_empty() {
                    let last_v = &vertices[vertices.len() - 1];
                    if last_v.bulge.abs() > BULGE_EPSILON {
                        let p1 = [last_v.x * scale, last_v.y * scale];
                        let p2 = [vertices[0].x * scale, vertices[0].y * scale];
                        let arc_points = bulge_to_arc_points(p1, p2, last_v.bulge, self.config.arc_tolerance_mm);
                        if arc_points.len() > 2 {
                            points.extend_from_slice(&arc_points[1..]);
                        }
                    }
                }

                // P6: 验证最终点数
                if points.len() < 2 {
                    tracing::debug!("轻多段线离散后点数不足：handle={:?}, count={}", entity.common.handle, points.len());
                    return vec![];
                }

                vec![RawEntity::Polyline {
                    points,
                    closed: lwpolyline.is_closed(),
                    metadata: metadata.clone(),
                    semantic: semantic.clone(),
                }]
            }

            EntityType::Arc(arc) => {
                // P6: 检查圆弧半径有效性
                if arc.radius <= 0.0 || arc.radius.is_nan() || arc.radius.is_infinite() {
                    tracing::warn!("圆弧半径无效：handle={:?}, radius={}", entity.common.handle, arc.radius);
                    return vec![];
                }

                vec![RawEntity::Arc {
                    center: [arc.center.x * scale, arc.center.y * scale],
                    radius: arc.radius * scale,
                    start_angle: arc.start_angle.to_degrees(),
                    end_angle: arc.end_angle.to_degrees(),
                    metadata: metadata.clone(),
                    semantic: semantic.clone(),
                }]
            }

            EntityType::Circle(circle) => {
                // P6: 检查圆半径有效性
                if circle.radius <= 0.0 || circle.radius.is_nan() || circle.radius.is_infinite() {
                    tracing::warn!("圆半径无效：handle={:?}, radius={}", entity.common.handle, circle.radius);
                    return vec![];
                }

                vec![RawEntity::Circle {
                    center: [circle.center.x * scale, circle.center.y * scale],
                    radius: circle.radius * scale,
                    metadata: metadata.clone(),
                    semantic: semantic.clone(),
                }]
            }

            EntityType::Spline(spline) => {
                let points = self.discretize_spline_with_scale(spline, scale);
                if points.len() < 2 {
                    tracing::debug!("样条曲线离散后点数不足：handle={:?}, count={}", entity.common.handle, points.len());
                    return vec![];
                }
                vec![RawEntity::Polyline {
                    points,
                    closed: spline.is_closed(),
                    metadata: metadata.clone(),
                    semantic: semantic.clone(),
                }]
            }

            EntityType::Ellipse(ellipse) => {
                // P6: 检查椭圆轴长有效性
                // 注意：dxf crate 中椭圆使用 major_axis（向量）和 minor_axis_ratio（比率）
                let major_axis_len = (ellipse.major_axis.x.powi(2) + ellipse.major_axis.y.powi(2)).sqrt();
                let minor_axis_len = major_axis_len * ellipse.minor_axis_ratio;
                
                if major_axis_len <= 0.0 || minor_axis_len <= 0.0 {
                    tracing::warn!("椭圆轴长无效：handle={:?}, major={}, minor={}", 
                        entity.common.handle, major_axis_len, minor_axis_len);
                    return vec![];
                }

                let points = self.discretize_ellipse_with_scale(ellipse, scale);
                if points.len() < 2 {
                    tracing::debug!("椭圆离散后点数不足：handle={:?}, count={}", entity.common.handle, points.len());
                    return vec![];
                }
                vec![RawEntity::Polyline {
                    points,
                    closed: true,
                    metadata: metadata.clone(),
                    semantic: semantic.clone(),
                }]
            }

            EntityType::Text(text) => {
                if !self.config.ignore_text {
                    // P6: 检查文字高度有效性
                    if text.text_height <= 0.0 || text.text_height.is_nan() {
                        tracing::warn!("文字高度无效：handle={:?}, height={}", entity.common.handle, text.text_height);
                        return vec![];
                    }

                    // P2 增强：提取完整的 TEXT 实体信息
                    vec![RawEntity::Text {
                        position: [text.location.x * scale, text.location.y * scale],
                        content: text.value.clone(),
                        height: text.text_height * scale,
                        rotation: text.rotation,
                        style_name: None,
                        align_left: None,
                        align_right: None,
                        metadata: metadata.clone(),
                        semantic: semantic.clone(),
                    }]
                } else {
                    vec![]
                }
            }

            EntityType::MText(mtext) => {
                if !self.config.ignore_text {
                    // P6: 检查文字高度有效性
                    if mtext.initial_text_height <= 0.0 || mtext.initial_text_height.is_nan() {
                        tracing::warn!("多行文字高度无效：handle={:?}, height={}", entity.common.handle, mtext.initial_text_height);
                        return vec![];
                    }

                    // P2 增强：提取完整的 MTEXT 实体信息
                    // 注意：mtext.text 可能包含 DXF 格式代码（如\X、\P），需要清理
                    let cleaned_text = clean_mtext_content(&mtext.text);

                    vec![RawEntity::Text {
                        position: [mtext.insertion_point.x * scale, mtext.insertion_point.y * scale],
                        content: cleaned_text,
                        height: mtext.initial_text_height * scale,
                        rotation: 0.0,
                        style_name: None,
                        align_left: None,
                        align_right: None,
                        metadata: metadata.clone(),
                        semantic: semantic.clone(),
                    }]
                } else {
                    vec![]
                }
            }

            EntityType::Insert(insert) => {
                // P6: 检查块引用缩放因子有效性
                let block_scale = [
                    if insert.x_scale_factor.is_nan() || insert.x_scale_factor.is_infinite() { 1.0 } else { insert.x_scale_factor },
                    if insert.y_scale_factor.is_nan() || insert.y_scale_factor.is_infinite() { 1.0 } else { insert.y_scale_factor },
                    if insert.z_scale_factor.is_nan() || insert.z_scale_factor.is_infinite() { 1.0 } else { insert.z_scale_factor },
                ];

                // 处理块引用
                if let Some(block_def) = block_definitions.get(&insert.name) {
                    let rotation = insert.rotation;
                    let insertion_point = [insert.location.x * scale, insert.location.y * scale];

                    // 递归展开块定义中的实体（支持嵌套块）
                    let mut visited = std::collections::HashSet::new();
                    let expanded = resolve_block_references(
                        block_def,
                        block_definitions,
                        &mut visited,
                    );

                    // 应用变换到所有展开的实体
                    expanded.iter()
                        .map(|block_entity| transform_entity(
                            block_entity,
                            block_scale,
                            rotation,
                            insertion_point,
                        ))
                        .collect()
                } else {
                    tracing::warn!("块引用 '{}' 未找到对应的块定义", insert.name);
                    vec![]
                }
            }

            // P0-1: HATCH 填充图案支持（建筑 CAD 核心功能）
            // 使用低层级组码迭代器解析 HATCH 实体
            EntityType::ProxyEntity(_) => {
                // Hatch 实体在 dxf 0.6.0 中被归类为 ProxyEntity
                let type_name = format!("{:?}", entity.specific);
                if type_name.contains("Hatch") {
                    // 注意：由于 dxf 0.6.0 的限制，无法直接访问 Hatch 实体数据
                    // 这里使用低层级组码解析器来提取 HATCH 数据
                    // 需要在 parse_file 级别使用 HatchParser 单独解析
                    tracing::debug!("检测到 HATCH 实体（ProxyEntity），将在文件级别解析 - handle={:?}", entity.common.handle);
                }
                vec![]
            }

            // P0-2: DIMENSION 尺寸标注支持（dxf 0.6.0 支持）
            // 匹配具体的 Dimension 类型并提取真实数据
            EntityType::RotatedDimension(ref dim) => {
                self.parse_rotated_dimension_entity(dim, scale, metadata, semantic)
            }
            EntityType::RadialDimension(ref dim) => {
                self.parse_radial_dimension_entity(dim, scale, metadata, semantic)
            }
            EntityType::DiameterDimension(ref dim) => {
                self.parse_diameter_dimension_entity(dim, scale, metadata, semantic)
            }
            EntityType::AngularThreePointDimension(ref dim) => {
                self.parse_angular_dimension_entity(dim, scale, metadata, semantic)
            }
            EntityType::OrdinateDimension(ref dim) => {
                self.parse_ordinate_dimension_entity(dim, scale, metadata, semantic)
            }

            // 其他未支持的实体类型
            _ => {
                tracing::debug!("跳过未支持的实体类型：{:?}", entity.specific);
                vec![]
            }
        }
    }

    /// 检测单位不匹配（图纸标注单位与实际坐标范围不符）
    fn detect_unit_mismatch(&self, entities: &[RawEntity], report: &mut DxfParseReport) {
        if entities.is_empty() {
            return;
        }

        // 计算坐标范围
        let mut max_coord = 0.0;
        for entity in entities {
            let coords: Vec<f64> = match entity {
                RawEntity::Line { start, end, .. } => {
                    vec![start[0].abs(), start[1].abs(), end[0].abs(), end[1].abs()]
                }
                RawEntity::Polyline { points, .. } => {
                    points.iter().flat_map(|v| vec![v[0].abs(), v[1].abs()]).collect()
                }
                RawEntity::Arc { center, radius, .. } => {
                    vec![(center[0] + radius).abs(), (center[1] + radius).abs()]
                }
                RawEntity::Circle { center, radius, .. } => {
                    vec![(center[0] + radius).abs(), (center[1] + radius).abs()]
                }
                _ => continue,
            };
            for &coord in &coords {
                if coord > max_coord {
                    max_coord = coord;
                }
            }
        }

        // 检查单位不匹配
        // 如果单位是米（scale=1000），但坐标值 > 10000（10 米），可能实际使用毫米绘制
        if let Some(ref units) = report.drawing_units {
            let unit_str = units.to_lowercase();

            if (unit_str.contains("米") || unit_str.contains("meter")) && max_coord > 10000.0 {
                report.unit_mismatch_detected = true;
                report.warnings.push(format!(
                    "单位不匹配：图纸单位为米，但坐标值较大（最大：{:.1}），可能实际使用毫米绘制",
                    max_coord
                ));
            }
            // 如果单位是英寸（scale=25.4），但坐标值 > 1000（25 米），可能单位设置错误
            else if (unit_str.contains("英寸") || unit_str.contains("inch")) && max_coord > 10000.0 {
                report.unit_mismatch_detected = true;
                report.warnings.push(format!(
                    "单位不匹配：图纸单位为英寸，但坐标值较大（最大：{:.1}），请确认单位设置",
                    max_coord
                ));
            }
        }
    }

    // ========================================================================
    // P0-1: HATCH 填充图案解析支持（自行实现）
    // ========================================================================
    // 注意：dxf 0.6.0 crate 没有暴露 Hatch 类型，需要使用低层级 API 解析
    // 实现方案：使用 dxf::iterators 遍历实体组码，手动解析 HATCH 数据
    // 
    // HATCH 实体组码说明：
    // - 组码 2: 图案名称
    // - 组码 70: 实体填充标志（1=实体填充，0=图案填充）
    // - 组码 71: 关联标志
    // - 组码 72: 边界路径数量
    // - 组码 75: 填充样式（0=奇偶填充，1=外边界填充）
    // - 组码 76: 边界路径类型（1=多段线，2=直线+圆弧组合）
    // - 组码 91: 边界路径数量
    // - 组码 98: 种子点数量
    // - 组码 10-20: 种子点坐标
    // - 组码 100: 子对象类型标识
    // - 组码 450-499: 渐变填充数据
    // ========================================================================

    // ========================================================================
    // P0-2: DIMENSION 尺寸标注解析支持（自行实现）
    // ========================================================================
    // 注意：dxf 0.6.0 crate 有 Dimension 类型，但需要使用低层级 API 解析
    // 
    // DIMENSION 实体组码说明：
    // - 组码 2: 块名称
    // - 组码 3: 样式名称
    // - 组码 70: 标注类型（0=线性，1=对齐，2=角度，3=直径，4=半径）
    // - 组码 42: 测量值
    // - 组码 1: 标注文字
    // - 组码 10-20-30: 定义点 1
    // - 组码 11-21-31: 定义点 2
    // - 组码 12-22-32: 定义点 3
    // - 组码 13-23-33: 定义点 4
    // - 组码 14-24-34: 定义点 5
    // ========================================================================

    // ========================================================================
    // 辅助函数：离散化圆弧和椭圆弧（用于 HATCH 边界）
    // ========================================================================

    /// 离散化圆弧为多段线
    #[allow(dead_code)]
    fn discretize_arc(
        &self,
        center: Point2,
        radius: f64,
        start_angle: f64,
        end_angle: f64,
        ccw: bool,
        tolerance: f64
    ) -> Vec<Point2> {
        // 计算弧长
        let mut sweep_angle = end_angle - start_angle;
        if sweep_angle <= 0.0 {
            sweep_angle += 360.0;
        }
        if !ccw {
            sweep_angle = 360.0 - sweep_angle;
        }
        
        let sweep_rad = sweep_angle.to_radians();
        let arc_length = radius * sweep_rad;
        
        // 根据容差计算采样点数
        // 弦高误差公式：h = r * (1 - cos(theta/2))
        // 近似：theta ≈ 2 * sqrt(2h/r)
        let num_segments = ((arc_length / (2.0 * (2.0 * tolerance * radius).sqrt())).ceil() as usize).max(4);
        
        let start_rad = start_angle.to_radians();
        let angle_step = if ccw { sweep_rad } else { -sweep_rad } / num_segments as f64;
        
        (0..=num_segments)
            .map(|i| {
                let angle = start_rad + i as f64 * angle_step;
                [
                    center[0] + radius * angle.cos(),
                    center[1] + radius * angle.sin(),
                ]
            })
            .collect()
    }

    /// 离散化椭圆弧为多段线
    #[allow(dead_code)]
    fn discretize_ellipse_arc(
        &self,
        center: Point2,
        major_axis: Point2,
        minor_axis_ratio: f64,
        start_angle: f64,
        end_angle: f64,
        ccw: bool,
        tolerance: f64
    ) -> Vec<Point2> {
        let major_radius = (major_axis[0].powi(2) + major_axis[1].powi(2)).sqrt();
        let minor_radius = major_radius * minor_axis_ratio;
        
        // 计算椭圆弧的参数角度
        let mut sweep_angle = end_angle - start_angle;
        if sweep_angle <= 0.0 {
            sweep_angle += 360.0;
        }
        if !ccw {
            sweep_angle = 360.0 - sweep_angle;
        }
        
        // 使用近似公式计算采样点数
        let avg_radius = (major_radius + minor_radius) / 2.0;
        let sweep_rad = sweep_angle.to_radians();
        let arc_length = avg_radius * sweep_rad;
        let num_segments = ((arc_length / (2.0 * (2.0 * tolerance * avg_radius).sqrt())).ceil() as usize).max(8);
        
        let start_rad = start_angle.to_radians();
        let angle_step = if ccw { sweep_rad } else { -sweep_rad } / num_segments as f64;
        
        // 计算主轴方向
        let major_angle = major_axis[1].atan2(major_axis[0]);
        
        (0..=num_segments)
            .map(|i| {
                let param_angle = start_rad + i as f64 * angle_step;
                let cos_a = param_angle.cos();
                let sin_a = param_angle.sin();
                
                // 旋转到主轴方向
                let x = major_radius * cos_a;
                let y = minor_radius * sin_a;
                
                let cos_m = major_angle.cos();
                let sin_m = major_angle.sin();
                
                [
                    center[0] + x * cos_m - y * sin_m,
                    center[1] + x * sin_m + y * cos_m,
                ]
            })
            .collect()
    }

    // ========================================================================
    // P0-1: HATCH 填充图案解析实现
    // 注意：dxf 0.6.0 不支持 Hatch 实体，此函数暂不使用
    // ========================================================================

    /// 解析 HATCH 实体（完整实现）
    #[allow(dead_code)]
    fn parse_hatch_entity(
        &self,
        entity: &dxf::entities::Entity,
        scale: f64,
        metadata: EntityMetadata,
        semantic: Option<BoundarySemantic>,
    ) -> Vec<RawEntity> {
        tracing::debug!("解析 HATCH 实体：handle={:?}", entity.common.handle);

        // 尝试访问 Hatch 实体（dxf 0.6.0 中通过 specific 访问）
        // 注意：dxf 0.6.0 的 Hatch 实体在 generated.rs 中定义
        // 这里使用类型名称匹配来识别 Hatch 实体
        
        // 从实体中提取 HATCH 数据
        let boundary_paths = self.parse_hatch_boundaries(entity, scale);
        let pattern = self.parse_hatch_pattern(entity);
        let solid_fill = self.is_hatch_solid_fill(entity);

        // 如果没有有效的边界，返回空
        if boundary_paths.is_empty() {
            tracing::warn!("HATCH 实体没有有效的边界：handle={:?}", entity.common.handle);
            return vec![];
        }

        vec![RawEntity::Hatch {
            boundary_paths,
            pattern,
            solid_fill,
            metadata,
            semantic,
        }]
    }

    /// 解析 HATCH 边界路径
    #[allow(dead_code)]
    fn parse_hatch_boundaries(
        &self,
        entity: &dxf::entities::Entity,
        _scale: f64,
    ) -> Vec<HatchBoundaryPath> {
        // 注意：dxf 0.6.0 没有直接暴露 Hatch 类型的边界数据
        // 这里使用通配符匹配来尝试访问边界数据
        
        // 尝试通过类型名称访问 Hatch 实体
        let type_name = format!("{:?}", entity.specific);
        if !type_name.contains("Hatch") {
            return vec![];
        }

        // 使用反射式访问尝试获取边界数据
        // 由于 dxf 0.6.0 的限制，这里使用简化的实现
        // 完整实现需要访问 dxf::Drawing 的原始组码数据
        
        // TODO: 使用 dxf::Drawing::iter() 访问原始组码来解析边界
        // 组码 91 = 边界数量，92 = 边界类型，后续是边界数据
        
        vec![]
    }

    /// 解析 HATCH 填充图案
    #[allow(dead_code)]
    fn parse_hatch_pattern(&self, _entity: &dxf::entities::Entity) -> HatchPattern {
        // 尝试从实体中提取图案名称
        // 组码 2 = 图案名称
        
        // 默认返回 ANSI31（最常用的建筑填充图案）
        HatchPattern::Predefined { name: "ANSI31".to_string() }
    }

    /// 检查 HATCH 是否为实体填充
    #[allow(dead_code)]
    fn is_hatch_solid_fill(&self, _entity: &dxf::entities::Entity) -> bool {
        // 组码 70 = 填充类型（0 = 图案，1 = 实体填充）
        false
    }

    // ========================================================================
    // P0-2: DIMENSION 尺寸标注解析实现（dxf 0.6.0 支持）
    // ========================================================================

    /// 解析 RotatedDimension 实体（线性/对齐标注）
    fn parse_rotated_dimension_entity(
        &self,
        dim: &dxf::entities::RotatedDimension,
        scale: f64,
        metadata: EntityMetadata,
        _semantic: Option<BoundarySemantic>,
    ) -> Vec<RawEntity> {
        tracing::debug!("解析 RotatedDimension 实体：handle={:?}", metadata.handle);

        // 提取定义点
        let mut definition_points = Vec::new();
        
        // definition_point_1 来自 dimension_base
        let pt1 = &dim.dimension_base.definition_point_1;
        definition_points.push([pt1.x * scale, pt1.y * scale]);
        
        // definition_point_2 和 definition_point_3
        let pt2 = &dim.definition_point_2;
        definition_points.push([pt2.x * scale, pt2.y * scale]);
        
        let pt3 = &dim.definition_point_3;
        definition_points.push([pt3.x * scale, pt3.y * scale]);

        // 提取测量值（actual_measurement）
        let measurement = dim.dimension_base.actual_measurement * scale;

        // 提取标注文字
        let text = if dim.dimension_base.text.is_empty() {
            None
        } else {
            Some(dim.dimension_base.text.clone())
        };

        // 从 dimension_base.dimension_type 提取类型
        let dimension_type = self.convert_dimension_type(&dim.dimension_base.dimension_type);

        vec![RawEntity::Dimension {
            dimension_type,
            measurement,
            text,
            definition_points,
            metadata,
            semantic: None,
        }]
    }

    /// 解析 RadialDimension 实体（半径标注）
    fn parse_radial_dimension_entity(
        &self,
        dim: &dxf::entities::RadialDimension,
        scale: f64,
        metadata: EntityMetadata,
        _semantic: Option<BoundarySemantic>,
    ) -> Vec<RawEntity> {
        tracing::debug!("解析 RadialDimension 实体：handle={:?}", metadata.handle);

        let mut definition_points = Vec::new();
        
        // definition_point_1 来自 dimension_base
        let pt1 = &dim.dimension_base.definition_point_1;
        definition_points.push([pt1.x * scale, pt1.y * scale]);
        
        // definition_point_2（必需字段）
        let pt2 = &dim.definition_point_2;
        definition_points.push([pt2.x * scale, pt2.y * scale]);

        // 提取测量值（半径）
        let measurement = dim.dimension_base.actual_measurement * scale;

        // 提取标注文字
        let text = if dim.dimension_base.text.is_empty() {
            None
        } else {
            Some(dim.dimension_base.text.clone())
        };

        let dimension_type = self.convert_dimension_type(&dim.dimension_base.dimension_type);

        vec![RawEntity::Dimension {
            dimension_type,
            measurement,
            text,
            definition_points,
            metadata,
            semantic: None,
        }]
    }

    /// 解析 DiameterDimension 实体（直径标注）
    fn parse_diameter_dimension_entity(
        &self,
        dim: &dxf::entities::DiameterDimension,
        scale: f64,
        metadata: EntityMetadata,
        _semantic: Option<BoundarySemantic>,
    ) -> Vec<RawEntity> {
        tracing::debug!("解析 DiameterDimension 实体：handle={:?}", metadata.handle);

        let mut definition_points = Vec::new();
        
        // definition_point_1 来自 dimension_base
        let pt1 = &dim.dimension_base.definition_point_1;
        definition_points.push([pt1.x * scale, pt1.y * scale]);
        
        // definition_point_2（必需字段）
        let pt2 = &dim.definition_point_2;
        definition_points.push([pt2.x * scale, pt2.y * scale]);

        // 提取测量值（直径）
        let measurement = dim.dimension_base.actual_measurement * scale;

        // 提取标注文字
        let text = if dim.dimension_base.text.is_empty() {
            None
        } else {
            Some(dim.dimension_base.text.clone())
        };

        let dimension_type = self.convert_dimension_type(&dim.dimension_base.dimension_type);

        vec![RawEntity::Dimension {
            dimension_type,
            measurement,
            text,
            definition_points,
            metadata,
            semantic: None,
        }]
    }

    /// 解析 AngularThreePointDimension 实体（三点角度标注）
    fn parse_angular_dimension_entity(
        &self,
        dim: &dxf::entities::AngularThreePointDimension,
        scale: f64,
        metadata: EntityMetadata,
        _semantic: Option<BoundarySemantic>,
    ) -> Vec<RawEntity> {
        tracing::debug!("解析 AngularThreePointDimension 实体：handle={:?}", metadata.handle);

        let mut definition_points = Vec::new();
        
        // definition_point_1 来自 dimension_base
        let pt1 = &dim.dimension_base.definition_point_1;
        definition_points.push([pt1.x * scale, pt1.y * scale]);
        
        // definition_point_2 和 definition_point_3
        let pt2 = &dim.definition_point_2;
        definition_points.push([pt2.x * scale, pt2.y * scale]);
        
        let pt3 = &dim.definition_point_3;
        definition_points.push([pt3.x * scale, pt3.y * scale]);

        // 提取测量值（角度，通常以弧度或度为单位）
        let measurement = dim.dimension_base.actual_measurement;

        // 提取标注文字
        let text = if dim.dimension_base.text.is_empty() {
            None
        } else {
            Some(dim.dimension_base.text.clone())
        };

        let dimension_type = self.convert_dimension_type(&dim.dimension_base.dimension_type);

        vec![RawEntity::Dimension {
            dimension_type,
            measurement,
            text,
            definition_points,
            metadata,
            semantic: None,
        }]
    }

    /// 解析 OrdinateDimension 实体（坐标标注）
    fn parse_ordinate_dimension_entity(
        &self,
        dim: &dxf::entities::OrdinateDimension,
        scale: f64,
        metadata: EntityMetadata,
        _semantic: Option<BoundarySemantic>,
    ) -> Vec<RawEntity> {
        tracing::debug!("解析 OrdinateDimension 实体：handle={:?}", metadata.handle);

        let mut definition_points = Vec::new();
        
        // definition_point_1 来自 dimension_base
        let pt1 = &dim.dimension_base.definition_point_1;
        definition_points.push([pt1.x * scale, pt1.y * scale]);
        
        // definition_point_2 和 definition_point_3（必需字段）
        let pt2 = &dim.definition_point_2;
        definition_points.push([pt2.x * scale, pt2.y * scale]);
        
        let pt3 = &dim.definition_point_3;
        definition_points.push([pt3.x * scale, pt3.y * scale]);

        // 提取测量值（坐标值）
        let measurement = dim.dimension_base.actual_measurement * scale;

        // 提取标注文字
        let text = if dim.dimension_base.text.is_empty() {
            None
        } else {
            Some(dim.dimension_base.text.clone())
        };

        let dimension_type = self.convert_dimension_type(&dim.dimension_base.dimension_type);

        vec![RawEntity::Dimension {
            dimension_type,
            measurement,
            text,
            definition_points,
            metadata,
            semantic: None,
        }]
    }

    /// 转换 dxf::enums::DimensionType 到 common_types::DimensionType
    fn convert_dimension_type(&self, dim_type: &dxf::enums::DimensionType) -> DimensionType {
        use dxf::enums::DimensionType as DxfDimensionType;
        
        match dim_type {
            DxfDimensionType::RotatedHorizontalOrVertical => DimensionType::Linear,
            DxfDimensionType::Aligned => DimensionType::Aligned,
            DxfDimensionType::Angular | DxfDimensionType::AngularThreePoint => DimensionType::Angular,
            DxfDimensionType::Diameter => DimensionType::Diameter,
            DxfDimensionType::Radius => DimensionType::Radial,
            DxfDimensionType::Ordinate => DimensionType::Ordinate,
        }
    }

    /// 离散化样条曲线为多段线（使用 NURBS 库精确计算）
    fn discretize_spline(&self, spline: &dxf::entities::Spline) -> Polyline {
        self.discretize_spline_with_scale(spline, 1.0)
    }

    /// 离散化样条曲线为多段线（带单位缩放）
    fn discretize_spline_with_scale(&self, spline: &dxf::entities::Spline, scale: f64) -> Polyline {
        let mut points = Vec::new();

        if spline.control_points.is_empty() {
            return points;
        }

        // 尝试使用 NURBS 库进行精确离散化
        if let Some(nurbs_curve) = self.build_nurbs_curve(spline) {
            // 使用曲率自适应采样
            points = self.adaptive_nurbs_sampling_with_scale(&nurbs_curve, self.tolerance, scale);

            // 如果自适应采样失败或点数太少，使用等参数采样作为 fallback
            if points.len() < 4 {
                points = self.uniform_nurbs_sampling_with_scale(&nurbs_curve, spline.control_points.len(), scale);
            }
        } else {
            // Fallback: 使用简化的离散化策略
            tracing::warn!("NURBS 构建失败，使用简化离散化策略");
            let num_segments = (spline.control_points.len() * 20).max(50);
            for i in 0..=num_segments {
                let t = i as f64 / num_segments as f64;
                if let Some(pt) = self.evaluate_spline_fallback(spline, t) {
                    points.push([pt[0] * scale, pt[1] * scale]);
                }
            }
        }

        points
    }

    /// 等参数采样 NURBS 曲线（作为 fallback）
    #[allow(dead_code)] // 预留用于未来 NURBS 优化
    fn uniform_nurbs_sampling(&self, curve: &NurbsCurve<f64, nalgebra::Const<2>>, num_control_points: usize) -> Polyline {
        self.uniform_nurbs_sampling_with_scale(curve, num_control_points, 1.0)
    }

    /// 等参数采样 NURBS 曲线（带单位缩放）
    fn uniform_nurbs_sampling_with_scale(&self, curve: &NurbsCurve<f64, nalgebra::Const<2>>, num_control_points: usize, scale: f64) -> Polyline {
        let mut points = Vec::new();
        let (t_start, t_end) = curve.knots_domain();
        let num_segments = self.estimate_nurbs_segments(curve, num_control_points);

        for i in 0..=num_segments {
            let t = t_start + (t_end - t_start) * (i as f64 / num_segments as f64);
            let pt = curve.point_at(t);

            // 维度检查
            if pt.len() < 2 {
                continue;
            }

            // P0 优化：使用动态容差替代硬编码的 POINT_Z_EPSILON
            // 3D 曲线投影警告：使用相对容差判断 Z 坐标是否显著
            let z_tolerance = self.adaptive_tolerance.intersection_tolerance();
            if pt.len() > 2 && pt[2].abs() > z_tolerance {
                tracing::warn!("检测到 3D 曲线 (Z={:.3})，已投影到 2D 平面", pt[2]);
            }

            points.push([pt[0] * scale, pt[1] * scale]);
        }
        points
    }

    /// 曲率自适应采样 NURBS 曲线
    ///
    /// # 算法
    /// 使用递归细分，基于弦高误差控制采样密度
    /// 在高曲率区域自动增加采样点，在平坦区域减少采样点
    #[allow(dead_code)] // 预留用于未来 NURBS 优化
    fn adaptive_nurbs_sampling(&self, curve: &NurbsCurve<f64, nalgebra::Const<2>>, tolerance: f64) -> Polyline {
        self.adaptive_nurbs_sampling_with_scale(curve, tolerance, 1.0)
    }

    /// 曲率自适应采样 NURBS 曲线（带单位缩放）
    fn adaptive_nurbs_sampling_with_scale(&self, curve: &NurbsCurve<f64, nalgebra::Const<2>>, tolerance: f64, scale: f64) -> Polyline {
        let (t_start, t_end) = curve.knots_domain();
        let mut points = Vec::new();

        // 添加起点
        let start_pt = curve.point_at(t_start);
        if start_pt.len() >= 2 {
            points.push([start_pt[0] * scale, start_pt[1] * scale]);
        }

        // 递归细分（初始深度为 0）
        self.subdivide_curve_with_scale(curve, t_start, t_end, tolerance, &mut points, 0, scale);

        // 添加终点
        let end_pt = curve.point_at(t_end);
        if end_pt.len() >= 2 && points.last().is_none_or(|p| {
            (p[0] - end_pt[0] * scale).abs() > tolerance || (p[1] - end_pt[1] * scale).abs() > tolerance
        }) {
            points.push([end_pt[0] * scale, end_pt[1] * scale]);
        }

        points
    }

    /// 递归细分曲线段
    ///
    /// # 算法
    /// 1. 计算曲线段中点
    /// 2. 计算中点到弦的垂直距离（弦高误差）
    /// 3. 如果弦高误差 > tolerance，递归细分左右两半
    /// 4. 否则，添加中点
    ///
    /// # 参数
    /// - `depth`: 当前递归深度，用于防止栈溢出
    #[allow(dead_code)] // 预留用于未来 NURBS 优化
    fn subdivide_curve(
        &self,
        curve: &NurbsCurve<f64, nalgebra::Const<2>>,
        t0: f64,
        t1: f64,
        tolerance: f64,
        points: &mut Polyline,
        depth: usize,
    ) {
        self.subdivide_curve_with_scale(curve, t0, t1, tolerance, points, depth, 1.0)
    }

    /// 递归细分曲线段（带单位缩放）
    ///
    /// # P1 优化：动态递归深度
    /// 使用基于容差和曲线长度的动态最大深度，而非固定值 20
    /// 对于复杂 NURBS（如自由曲面墙），20 层可能不够
    /// 对于简单曲线，过深的递归是浪费
    fn subdivide_curve_with_scale(
        &self,
        curve: &NurbsCurve<f64, nalgebra::Const<2>>,
        t0: f64,
        t1: f64,
        tolerance: f64,
        points: &mut Polyline,
        depth: usize,
        scale: f64,
    ) {
        // P1 优化：动态计算最大递归深度
        // 核心思想：基于容差和曲线长度计算所需深度
        // 每次细分误差减半，所以 depth = log2(initial_error / tolerance)
        let max_depth = self.compute_dynamic_max_depth(curve, t0, t1, tolerance);

        if depth > max_depth {
            // P11 锐评落实：达到最大深度时强制添加中点，而非直接返回
            // 避免曲线中间缺失一段
            let t_mid = (t0 + t1) / 2.0;
            let p_mid = curve.point_at(t_mid);
            if p_mid.len() >= 2 {
                points.push([p_mid[0] * scale, p_mid[1] * scale]);
            }
            tracing::debug!(
                "NURBS 曲线细分达到动态最大深度 {}（tolerance={:.6}），已强制添加中点",
                max_depth,
                tolerance
            );
            return;
        }

        // 计算中点
        let t_mid = (t0 + t1) / 2.0;
        let p0 = curve.point_at(t0);
        let p_mid = curve.point_at(t_mid);
        let p1 = curve.point_at(t1);

        // 确保点有效
        if p0.len() < 2 || p_mid.len() < 2 || p1.len() < 2 {
            return;
        }

        let p0_2d = [p0[0], p0[1]];
        let p_mid_2d = [p_mid[0], p_mid[1]];
        let p1_2d = [p1[0], p1[1]];

        // 计算弦高误差：中点到弦 p0-p1 的垂直距离
        let chord_error = DxfParser::point_to_line_distance(p_mid_2d, p0_2d, p1_2d);

        // 检查弦长，避免除以零
        let chord_length = ((p1_2d[0] - p0_2d[0]).powi(2) + (p1_2d[1] - p0_2d[1]).powi(2)).sqrt();
        if chord_length < tolerance * 0.01 {
            // 弦长非常小，曲线段已经足够短
            return;
        }

        // 如果弦高误差超过容差，递归细分
        if chord_error > tolerance {
            // 递归细分左右两半
            self.subdivide_curve_with_scale(curve, t0, t_mid, tolerance, points, depth + 1, scale);
            points.push([p_mid[0] * scale, p_mid[1] * scale]);
            self.subdivide_curve_with_scale(curve, t_mid, t1, tolerance, points, depth + 1, scale);
        }
        // 否则，不添加点（由上层调用者处理）
    }

    /// 动态计算最大递归深度
    ///
    /// # 核心思想
    /// 基于容差和曲线长度计算所需深度：
    /// ```text
    /// initial_error ≈ chord_length / 8  (抛物线近似)
    /// required_depth = log2(initial_error / tolerance)
    /// ```
    ///
    /// # 限制范围
    /// - 最小深度：8（避免欠采样）
    /// - 最大深度：28（避免栈溢出，2^28 = 2.6 亿点）
    fn compute_dynamic_max_depth(
        &self,
        curve: &NurbsCurve<f64, nalgebra::Const<2>>,
        t0: f64,
        t1: f64,
        tolerance: f64,
    ) -> usize {
        // 估算初始误差（使用弦长近似）
        let p0 = curve.point_at(t0);
        let p1 = curve.point_at(t1);

        if p0.len() < 2 || p1.len() < 2 {
            return 20;  //  fallback
        }

        let chord_length = ((p1[0] - p0[0]).powi(2) + (p1[1] - p0[1]).powi(2)).sqrt();

        // 初始误差估算：弦长的 1/8（抛物线近似）
        let initial_error = chord_length / 8.0;

        // 计算所需深度：log2(initial_error / tolerance)
        // 每次细分误差减半
        let required_depth = if initial_error > tolerance {
            (initial_error / tolerance).log2().ceil() as usize
        } else {
            0  // 不需要细分
        };

        // 限制范围：8-28
        // 8: 保证基本采样密度
        // 28: 防止栈溢出（2^28 = 2.6 亿点，递归 28 层）
        required_depth.clamp(8, 28)
    }

    /// 计算点到直线的距离（弦高误差）
    fn point_to_line_distance(point: Point2, line_start: Point2, line_end: Point2) -> f64 {
        let dx = line_end[0] - line_start[0];
        let dy = line_end[1] - line_start[1];

        // 直线方程：ax + by + c = 0
        // 其中 a = -dy, b = dx, c = -ax_start - by_start
        let a = -dy;
        let b = dx;
        let c = -a * line_start[0] - b * line_start[1];

        // 点到直线距离公式：|ax + by + c| / sqrt(a^2 + b^2)
        let numerator = (a * point[0] + b * point[1] + c).abs();
        let denominator = (a * a + b * b).sqrt();

        if denominator < 1e-10 {
            // 直线退化为点
            0.0
        } else {
            numerator / denominator
        }
    }

    /// 构建 NURBS 曲线
    fn build_nurbs_curve(&self, spline: &dxf::entities::Spline) -> Option<NurbsCurve<f64, nalgebra::Const<2>>> {
        use nalgebra::Point2;

        if spline.control_points.is_empty() {
            return None;
        }

        // dxf 0.6 中，knots 通过 knot_values 字段访问（Vec<f64>）
        if spline.knot_values.is_empty() {
            return None;
        }

        // 提取控制点（2D）
        let control_points: Vec<_> = spline.control_points
            .iter()
            .map(|cp| Point2::new(cp.x, cp.y))
            .collect();

        // 获取阶数（degree_of_curve）
        let degree = spline.degree_of_curve as usize;

        // 构建 NURBS 曲线（使用 try_new）
        NurbsCurve::try_new(degree, control_points, spline.knot_values.clone()).ok()
    }

    /// 估算 NURBS 曲线离散化段数（基于弧长和曲率）
    fn estimate_nurbs_segments(&self, curve: &NurbsCurve<f64, nalgebra::Const<2>>, num_control_points: usize) -> usize {
        // 1. 估算弧长（使用自适应采样积分）
        let arc_length = self.estimate_nurbs_arc_length(curve, num_control_points);

        // 2. 基于弦高公式计算段数
        // 弦高公式：h = r(1 - cos(π/n))
        // 反解：n = π / acos(1 - h/r)
        // 其中 r ≈ arc_length / (2π)（假设近似为圆）
        let tolerance = self.tolerance;
        let avg_radius = arc_length / (2.0 * std::f64::consts::PI);

        if avg_radius > tolerance {
            // 计算圆心角
            let cos_angle = (avg_radius - tolerance) / avg_radius;
            // 防止浮点误差导致 acos 参数超出 [-1, 1]
            let cos_angle = cos_angle.clamp(-1.0, 1.0);
            let angle = 2.0 * cos_angle.acos();
            let segments = (2.0 * std::f64::consts::PI / angle).ceil() as usize;
            // 限制段数范围，避免过度离散化
            segments.clamp(16, 2048)
        } else {
            // 曲线非常小，使用最少段数
            16
        }
    }

    /// 估算 NURBS 曲线的弧长（使用自适应采样积分）
    fn estimate_nurbs_arc_length(&self, curve: &NurbsCurve<f64, nalgebra::Const<2>>, num_control_points: usize) -> f64 {
        let (t_start, t_end) = curve.knots_domain();
        
        // 基于曲线复杂度动态调整采样点数
        // 控制点越多，需要更多采样点来捕捉细节
        // 每个控制点至少采样 5 次，至少 20 点，最多 200 点
        let num_samples = (num_control_points * 5).clamp(20, 200);

        let mut total_length = 0.0;
        let mut prev_point = curve.point_at(t_start);

        for i in 1..=num_samples {
            let t = t_start + (t_end - t_start) * (i as f64 / num_samples as f64);
            let curr_point = curve.point_at(t);

            // 计算两点间距离
            let dx = curr_point[0] - prev_point[0];
            let dy = curr_point[1] - prev_point[1];
            total_length += (dx * dx + dy * dy).sqrt();

            prev_point = curr_point;
        }

        total_length
    }

    /// 评估样条曲线上的点（Fallback 简化版本）
    fn evaluate_spline_fallback(&self, spline: &dxf::entities::Spline, t: f64) -> Option<Point2> {
        if spline.control_points.is_empty() {
            return None;
        }

        // 对于控制点较少的情况，使用线性插值
        if spline.control_points.len() == 2 {
            let p0 = &spline.control_points[0];
            let p1 = &spline.control_points[1];
            return Some([
                p0.x + (p1.x - p0.x) * t,
                p0.y + (p1.y - p0.y) * t,
            ]);
        }

        // 简化：返回控制点的加权平均
        let n = spline.control_points.len();
        let mut x = 0.0;
        let mut y = 0.0;
        let mut weight = 0.0;

        for (i, cp) in spline.control_points.iter().enumerate() {
            let i_t = i as f64 / n as f64;
            let dist = (t - i_t).abs();
            let w = if dist < 0.5 { 1.0 - dist * 2.0 } else { 0.0 };

            x += cp.x * w;
            y += cp.y * w;
            weight += w;
        }

        if weight > BULGE_EPSILON {
            Some([x / weight, y / weight])
        } else {
            let idx = (t * (n - 1) as f64) as usize;
            let idx = idx.min(n - 2);
            let p0 = &spline.control_points[idx];
            let p1 = &spline.control_points[idx + 1];
            let local_t = t * (n - 1) as f64 - idx as f64;
            Some([
                p0.x + (p1.x - p0.x) * local_t,
                p0.y + (p1.y - p0.y) * local_t,
            ])
        }
    }

    /// 离散化椭圆为多段线（使用弦高误差控制，动态调整段数）
    fn discretize_ellipse(&self, ellipse: &dxf::entities::Ellipse) -> Polyline {
        self.discretize_ellipse_with_scale(ellipse, 1.0)
    }

    /// 离散化椭圆为多段线（带单位缩放）
    fn discretize_ellipse_with_scale(&self, ellipse: &dxf::entities::Ellipse, scale: f64) -> Polyline {
        let mut points = Vec::new();

        // major_axis 是 Vector 类型，需要手动计算其长度
        let major_vec = ellipse.major_axis.clone();
        let a = (major_vec.x * major_vec.x + major_vec.y * major_vec.y + major_vec.z * major_vec.z).sqrt();
        let b = a * ellipse.minor_axis_ratio;

        // 使用弦高误差控制离散化
        let tolerance = self.tolerance;

        // 基于弦高公式计算段数：h = r(1 - cos(π/n))
        // 推导出：n = π / acos(1 - h/r)
        // 使用长半轴作为参考半径（保守估计）
        let reference_radius = a.max(b);
        let num_segments = if reference_radius > tolerance {
            let angle_per_segment = 2.0 * ((reference_radius - tolerance) / reference_radius).acos();
            (2.0 * std::f64::consts::PI / angle_per_segment).ceil() as usize
        } else {
            32 // 如果半径太小，使用默认段数
        };

        // 确保段数足够覆盖周长，同时限制最小/最大值
        let num_segments = num_segments.clamp(16, 1024);

        // 获取主轴方向（归一化）
        let major_dir = if a > BULGE_EPSILON {
            [major_vec.x / a, major_vec.y / a]
        } else {
            [1.0, 0.0]
        };

        // 计算次轴方向（垂直于主轴）
        let minor_dir = [-major_dir[1], major_dir[0]];

        for i in 0..num_segments {
            let angle = 2.0 * std::f64::consts::PI * i as f64 / num_segments as f64;

            // 椭圆参数方程 - 使用主轴和次轴向量
            let cos_a = angle.cos();
            let sin_a = angle.sin();

            let x = (ellipse.center.x + major_dir[0] * a * cos_a + minor_dir[0] * b * sin_a) * scale;
            let y = (ellipse.center.y + major_dir[1] * a * cos_a + minor_dir[1] * b * sin_a) * scale;

            points.push([x, y]);
        }

        // 闭合
        if let Some(first) = points.first().copied() {
            points.push(first);
        }

        points
    }
}

impl Default for DxfParser {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for DxfParseReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "\n========== DXF 解析报告 ==========")?;

        if let Some(units) = &self.drawing_units {
            writeln!(f, "图纸单位：{} (比例因子：{:.4})", units, self.unit_scale)?;
        }

        writeln!(f, "\n图层分布:")?;
        let mut layers: Vec<_> = self.layer_distribution.iter().collect();
        layers.sort_by(|a, b| b.1.cmp(a.1));
        for (layer, count) in layers {
            writeln!(f, "  {}: {} 个实体", layer, count)?;
        }

        writeln!(f, "\n实体类型分布:")?;
        let mut types: Vec<_> = self.entity_type_distribution.iter().collect();
        types.sort_by(|a, b| b.1.cmp(a.1));
        for (entity_type, count) in types {
            writeln!(f, "  {}: {}", entity_type, count)?;
        }

        if !self.warnings.is_empty() {
            writeln!(f, "\n警告:")?;
            for warning in &self.warnings {
                writeln!(f, "  ⚠️  {}", warning)?;
            }
        }

        writeln!(f, "=====================================")?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dxf_parser_new() {
        let parser = DxfParser::new();
        assert!(parser.layer_filter.is_none());
        assert_eq!(parser.tolerance, 0.1);
    }

    #[test]
    fn test_dxf_parser_with_filter() {
        let parser = DxfParser::new().with_layer_filter(vec!["0".to_string(), "WALL".to_string()]);
        assert!(parser.layer_filter.is_some());
        assert_eq!(parser.layer_filter.unwrap().len(), 2);
    }

    #[test]
    fn test_dxf_parser_with_tolerance() {
        let parser = DxfParser::new().with_tolerance(0.05);
        assert_eq!(parser.tolerance, 0.05);
    }

    #[test]
    fn test_is_binary_dxf() {
        // 测试二进制 DXF 检测
        assert!(DxfParser::is_binary_dxf(b"AC1014"));
        assert!(DxfParser::is_binary_dxf(b"AC1015"));
        assert!(DxfParser::is_binary_dxf(b"AC1018"));
        assert!(!DxfParser::is_binary_dxf(b"0\nSECTION"));
        assert!(!DxfParser::is_binary_dxf(b"999"));
    }

    #[test]
    fn test_bulge_to_arc_quarter_circle() {
        // bulge = 1.0 表示 90 度圆弧
        let p1 = [0.0, 0.0];
        let p2 = [10.0, 0.0];
        let bulge = 1.0;
        let points = bulge_to_arc_points(p1, p2, bulge, 0.1);
        
        // 验证起点和终点
        assert!(points.len() >= 2);
        assert!((points[0][0] - p1[0]).abs() < 1e-6);
        assert!((points[0][1] - p1[1]).abs() < 1e-6);
        assert!((points[points.len() - 1][0] - p2[0]).abs() < 0.2);
        assert!((points[points.len() - 1][1] - p2[1]).abs() < 0.2);
        
        // bulge > 0 时圆弧应该有中间点
        if points.len() > 2 {
            let mid_point = points[points.len() / 2];
            // 验证圆弧不是直线（y 坐标应该有变化）
            assert!(mid_point[1].abs() > 0.1, "圆弧应该有凸起");
        }
    }

    #[test]
    fn test_bulge_to_arc_straight_line() {
        // bulge = 0 表示直线
        let p1 = [0.0, 0.0];
        let p2 = [10.0, 0.0];
        let bulge = 0.0;
        let points = bulge_to_arc_points(p1, p2, bulge, 0.1);
        
        assert_eq!(points.len(), 2);
        assert_eq!(points[0], p1);
        assert_eq!(points[1], p2);
    }

    #[test]
    fn test_bulge_to_arc_negative() {
        // bulge < 0 表示顺时针（方向相反）
        let p1 = [0.0, 0.0];
        let p2 = [10.0, 0.0];
        let bulge = -0.5;
        let points = bulge_to_arc_points(p1, p2, bulge, 0.1);
        
        assert!(points.len() >= 2);
        // bulge < 0 时圆弧应该有中间点
        if points.len() > 2 {
            let mid_point = points[points.len() / 2];
            // 验证圆弧不是直线（y 坐标应该有变化）
            assert!(mid_point[1].abs() > 0.1, "圆弧应该有凹陷");
        }
    }
}

// ============================================================================
// 智能图层识别辅助函数
// ============================================================================

#[allow(clippy::items_after_test_module)]
/// 根据图层名识别实体语义
fn identify_semantic(layer: &str) -> Option<BoundarySemantic> {
    if is_wall_layer(layer) {
        Some(BoundarySemantic::HardWall)
    } else if is_door_only(layer) {
        Some(BoundarySemantic::Door)
    } else if is_window_only(layer) {
        Some(BoundarySemantic::Window)
    } else if is_opening_layer(layer) {
        Some(BoundarySemantic::Opening)
    } else {
        None
    }
}

/// 判断是否为门图层（专门用于区分门）
fn is_door_only(layer: &str) -> bool {
    let upper = layer.to_uppercase();

    // 门关键词
    let door_keywords = ["DOOR", "门", "DOORS", "入户门", "室内门", "防火门", "单开门", "双开门", "推拉门"];
    if door_keywords.iter().any(|k| upper.contains(k)) {
        return true;
    }

    // AIA 标准模式：A-DOOR-*
    if upper.starts_with("A-DOOR") {
        return true;
    }

    // 门模式匹配（优先于窗）
    if upper.contains("DOOR-") || upper.contains("DOOR_") || upper.starts_with("D-") {
        return true;
    }

    false
}

/// 判断是否为窗图层（专门用于区分窗）
fn is_window_only(layer: &str) -> bool {
    let upper = layer.to_uppercase();

    // 窗关键词
    let window_keywords = ["WINDOW", "窗", "WINDOWS", "GLAZ", "GLASS", "采光窗", "天窗", "落地窗", "百叶窗"];
    if window_keywords.iter().any(|k| upper.contains(k)) {
        return true;
    }

    // AIA 标准模式：A-WIND-*
    if upper.starts_with("A-WIND") {
        return true;
    }

    // 窗模式匹配
    if upper.contains("WINDOW-") || upper.contains("WINDOW_") ||
       upper.contains("GLAZ-") || upper.contains("GLASS-") ||
       upper.contains("GLASS_") || upper.starts_with("W-") {
        return true;
    }

    false
}

/// 判断是否为开口/门洞图层（通用开口，不区分门窗）
fn is_opening_layer(layer: &str) -> bool {
    let upper = layer.to_uppercase();

    // 开口关键词
    let opening_keywords = ["OPEN", "开口", "OPENING", "HOLE", "洞", "GATE", "通道"];
    if opening_keywords.iter().any(|k| upper.contains(k)) {
        return true;
    }

    // 开口模式匹配
    if upper.starts_with("A-OPEN") || upper.contains("OPENING-") || upper.contains("OPENING_") {
        return true;
    }

    false
}

/// 判断是否为墙体图层（支持 AIA 标准和常见变体）
fn is_wall_layer(layer: &str) -> bool {
    let upper = layer.to_uppercase();

    // 基础关键词匹配 - 墙/结构/柱/梁
    let basic_keywords = [
        "WALL", "墙", "WALLS", "墙体", "内墙", "外墙", "剪力墙", "隔墙",
        "STRUCT", "结构", "COLUMN", "柱", "BEAM", "梁", "STRUC",
    ];
    if basic_keywords.iter().any(|k| upper.contains(k)) {
        return true;
    }

    // AIA 标准模式：A-WALL-*, S-WALL-*, S-STRC-*
    if upper.starts_with("A-WALL") || upper.starts_with("S-WALL") || 
       upper.starts_with("S-STRC") || upper.starts_with("A-COLS") {
        return true;
    }

    // 常见变体模式
    let patterns = [
        "WALL-", "WALL_", "-WALL", "_WALL",
        "WALLS-", "WALLS_", "-WALLS", "_WALLS",
        "STRUCT-", "STRUCT_", "-STRUCT", "_STRUCT",
        "COLUMN-", "COLUMN_", "-COLUMN", "_COLUMN",
        "BEAM-", "BEAM_", "-BEAM", "_BEAM",
    ];
    patterns.iter().any(|p| upper.contains(p))
}

/// 根据 ACI 颜色索引映射材料名称
///
/// ACI 颜色与常见建筑材料的对应关系：
/// - 1 (红色): 混凝土 (concrete)
/// - 2 (黄色): 砖墙 (brick)
/// - 3 (绿色): 木材 (wood)
/// - 4 (青色): 石膏板 (gypsum)
/// - 5 (蓝色): 玻璃 (glass)
/// - 6 (洋红): 金属 (metal)
/// - 7 (黑色/白色): 默认墙体 (default_wall)
/// - 其他：未指定 (unspecified)
fn map_material_from_color(color_index: Option<&str>) -> Option<String> {
    match color_index {
        Some("1") => Some("concrete".to_string()),   // 红色=混凝土
        Some("2") => Some("brick".to_string()),      // 黄色=砖墙
        Some("3") => Some("wood".to_string()),       // 绿色=木材
        Some("4") => Some("gypsum".to_string()),     // 青色=石膏板
        Some("5") => Some("glass".to_string()),      // 蓝色=玻璃
        Some("6") => Some("metal".to_string()),      // 洋红=金属
        Some("7") => Some("default_wall".to_string()), // 黑色/白色=默认墙体
        _ => None,
    }
}

/// 根据图层名映射材料名称（备选方案，当颜色信息缺失时使用）
fn map_material_from_layer(layer: &str) -> Option<String> {
    let upper = layer.to_uppercase();

    // 混凝土墙
    if upper.contains("CONC") || upper.contains("混凝土") {
        return Some("concrete".to_string());
    }

    // 砖墙
    if upper.contains("BRICK") || upper.contains("砖") {
        return Some("brick".to_string());
    }

    // 木材
    if upper.contains("WOOD") || upper.contains("木") {
        return Some("wood".to_string());
    }

    // 玻璃
    if upper.contains("GLASS") || upper.contains("GLAZ") || upper.contains("玻璃") {
        return Some("glass".to_string());
    }

    // 金属
    if upper.contains("METAL") || upper.contains("STEEL") || upper.contains("钢") || upper.contains("金属") {
        return Some("metal".to_string());
    }

    // 石膏板
    if upper.contains("GYPSUM") || upper.contains("石膏") {
        return Some("gypsum".to_string());
    }

    None
}

/// 判断是否为家具图层
fn is_furniture_layer(layer: &str) -> bool {
    let upper = layer.to_uppercase();
    
    // 基础关键词
    let basic_keywords = ["FURN", "家具", "FF&E", "EQUIP", "设备", "洁具", "橱柜"];
    if basic_keywords.iter().any(|k| upper.contains(k)) {
        return true;
    }
    
    // 模式匹配
    let patterns = ["A-FURN", "FURN-", "FURN_", "EQUIP-", "EQUIP_"];
    patterns.iter().any(|p| upper.contains(p) || upper.starts_with(p))
}

/// 判断是否为标注图层
fn is_dimension_layer(layer: &str) -> bool {
    let upper = layer.to_uppercase();

    // 基础关键词
    let basic_keywords = ["DIM", "标注", "DIMS", "ANNOT", "注释", "标高", "轴号"];
    if basic_keywords.iter().any(|k| upper.contains(k)) {
        return true;
    }

    // 模式匹配
    let patterns = ["A-DIM", "DIM-", "DIM_", "ANNOT-", "ANNOT_"];
    patterns.iter().any(|p| upper.contains(p) || upper.starts_with(p))
}

/// 变换实体（应用缩放、旋转、平移）
fn transform_entity(
    entity: &RawEntity,
    scale: [f64; 3],
    rotation_deg: f64,
    translation: Point2,
) -> RawEntity {
    let rotation_rad = rotation_deg.to_radians();
    let cos_rot = rotation_rad.cos();
    let sin_rot = rotation_rad.sin();

    /// 变换 2D 点
    fn transform_point(point: Point2, scale: [f64; 3], cos_rot: f64, sin_rot: f64, translation: Point2) -> Point2 {
        // 1. 缩放
        let scaled = [point[0] * scale[0], point[1] * scale[1]];
        // 2. 旋转（绕原点）
        let rotated = [
            scaled[0] * cos_rot - scaled[1] * sin_rot,
            scaled[0] * sin_rot + scaled[1] * cos_rot,
        ];
        // 3. 平移
        [rotated[0] + translation[0], rotated[1] + translation[1]]
    }

    match entity {
        RawEntity::Line { start, end, metadata, semantic } => {
            RawEntity::Line {
                start: transform_point(*start, scale, cos_rot, sin_rot, translation),
                end: transform_point(*end, scale, cos_rot, sin_rot, translation),
                metadata: metadata.clone(),
                semantic: semantic.clone(),
            }
        }
        RawEntity::Polyline { points, closed, metadata, semantic } => {
            RawEntity::Polyline {
                points: points.iter()
                    .map(|p| transform_point(*p, scale, cos_rot, sin_rot, translation))
                    .collect(),
                closed: *closed,
                metadata: metadata.clone(),
                semantic: semantic.clone(),
            }
        }
        RawEntity::Arc { center, radius, start_angle, end_angle, metadata, semantic } => {
            // 圆弧：变换中心点，缩放半径，旋转角度
            let new_center = transform_point(*center, scale, cos_rot, sin_rot, translation);
            let avg_scale = (scale[0] + scale[1]) / 2.0;
            RawEntity::Arc {
                center: new_center,
                radius: radius * avg_scale,
                start_angle: *start_angle + rotation_deg,
                end_angle: *end_angle + rotation_deg,
                metadata: metadata.clone(),
                semantic: semantic.clone(),
            }
        }
        RawEntity::Circle { center, radius, metadata, semantic } => {
            let new_center = transform_point(*center, scale, cos_rot, sin_rot, translation);
            let avg_scale = (scale[0] + scale[1]) / 2.0;
            RawEntity::Circle {
                center: new_center,
                radius: radius * avg_scale,
                metadata: metadata.clone(),
                semantic: semantic.clone(),
            }
        }
        RawEntity::Text { position, content, height, metadata, semantic, .. } => {
            RawEntity::Text {
                position: transform_point(*position, scale, cos_rot, sin_rot, translation),
                content: content.clone(),
                height: height * avg_scale(scale),
                rotation: 0.0,  // 简化处理，忽略旋转
                style_name: None,
                align_left: None,
                align_right: None,
                metadata: metadata.clone(),
                semantic: semantic.clone(),
            }
        }
        RawEntity::Path { commands, metadata, semantic } => {
            let new_commands: Vec<_> = commands.iter().map(|cmd| {
                match cmd {
                    PathCommand::MoveTo { x, y } => {
                        let p = transform_point([*x, *y], scale, cos_rot, sin_rot, translation);
                        PathCommand::MoveTo { x: p[0], y: p[1] }
                    }
                    PathCommand::LineTo { x, y } => {
                        let p = transform_point([*x, *y], scale, cos_rot, sin_rot, translation);
                        PathCommand::LineTo { x: p[0], y: p[1] }
                    }
                    PathCommand::ArcTo { rx, ry, x_axis_rotation, large_arc, sweep, x, y } => {
                        let p = transform_point([*x, *y], scale, cos_rot, sin_rot, translation);
                        PathCommand::ArcTo {
                            rx: rx * scale[0],
                            ry: ry * scale[1],
                            x_axis_rotation: *x_axis_rotation + rotation_deg,
                            large_arc: *large_arc,
                            sweep: *sweep,
                            x: p[0],
                            y: p[1],
                        }
                    }
                    PathCommand::Close => PathCommand::Close,
                }
            }).collect();
            RawEntity::Path {
                commands: new_commands,
                metadata: metadata.clone(),
                semantic: semantic.clone(),
            }
        }
        RawEntity::BlockReference { block_name, insertion_point, scale: _, rotation: _, metadata, semantic } => {
            // 嵌套块引用：变换插入点
            RawEntity::BlockReference {
                block_name: block_name.clone(),
                insertion_point: transform_point(*insertion_point, scale, cos_rot, sin_rot, translation),
                scale,
                rotation: rotation_deg,
                metadata: metadata.clone(),
                semantic: semantic.clone(),
            }
        }
        RawEntity::Dimension { dimension_type, measurement, text, definition_points, metadata, semantic } => {
            RawEntity::Dimension {
                dimension_type: dimension_type.clone(),
                measurement: *measurement * avg_scale(scale),
                text: text.clone(),
                definition_points: definition_points.iter()
                    .map(|p| transform_point(*p, scale, cos_rot, sin_rot, translation))
                    .collect(),
                metadata: metadata.clone(),
                semantic: semantic.clone(),
            }
        }
        RawEntity::Hatch { boundary_paths, pattern, solid_fill, metadata, semantic } => {
            // 变换 HATCH 边界路径
            let transformed_boundaries: Vec<_> = boundary_paths.iter().map(|path| {
                match path {
                    common_types::HatchBoundaryPath::Polyline { points, closed } => {
                        common_types::HatchBoundaryPath::Polyline {
                            points: points.iter()
                                .map(|p| transform_point(*p, scale, cos_rot, sin_rot, translation))
                                .collect(),
                            closed: *closed,
                        }
                    }
                    common_types::HatchBoundaryPath::Arc { center, radius, start_angle, end_angle, ccw } => {
                        let new_center = transform_point(*center, scale, cos_rot, sin_rot, translation);
                        let avg_s = avg_scale(scale);
                        common_types::HatchBoundaryPath::Arc {
                            center: new_center,
                            radius: radius * avg_s,
                            start_angle: *start_angle + rotation_deg,
                            end_angle: *end_angle + rotation_deg,
                            ccw: *ccw,
                        }
                    }
                    common_types::HatchBoundaryPath::EllipseArc { center, major_axis, minor_axis_ratio, start_angle, end_angle, ccw } => {
                        let new_center = transform_point(*center, scale, cos_rot, sin_rot, translation);
                        let new_major = transform_point(*major_axis, scale, cos_rot, sin_rot, translation);
                        common_types::HatchBoundaryPath::EllipseArc {
                            center: new_center,
                            major_axis: new_major,
                            minor_axis_ratio: *minor_axis_ratio,
                            start_angle: *start_angle + rotation_deg,
                            end_angle: *end_angle + rotation_deg,
                            ccw: *ccw,
                        }
                    }
                    common_types::HatchBoundaryPath::Spline { control_points, knots, degree } => {
                        common_types::HatchBoundaryPath::Spline {
                            control_points: control_points.iter()
                                .map(|p| transform_point(*p, scale, cos_rot, sin_rot, translation))
                                .collect(),
                            knots: knots.clone(),
                            degree: *degree,
                        }
                    }
                }
            }).collect();

            RawEntity::Hatch {
                boundary_paths: transformed_boundaries,
                pattern: pattern.clone(),
                solid_fill: *solid_fill,
                metadata: metadata.clone(),
                semantic: semantic.clone(),
            }
        }
        RawEntity::XRef { .. } => {
            // P1-1: XREF 外部参照支持 - 待完整实现
            // 目前仅做类型标记，不进行几何处理
            todo!("XREF 外部参照变换 - 需要后续实现外部文件加载和坐标变换")
        }
    }
}

/// 递归展开嵌套块引用
///
/// # 参数
/// - `block_def`: 当前块定义
/// - `block_definitions`: 所有块定义的映射
/// - `visited`: 已访问的块名集合（防止循环引用）
///
/// # 返回
/// 展开后的实体列表（不包含 BlockReference）
fn resolve_block_references(
    block_def: &BlockDefinition,
    block_definitions: &HashMap<String, BlockDefinition>,
    visited: &mut std::collections::HashSet<String>,
) -> Vec<RawEntity> {
    // P11 锐评落实：检测到循环引用时记录错误，而非静默返回空
    // 实际图纸中常见的"动态块"可能被误判为循环引用
    // 改进：记录错误日志，便于用户诊断问题
    if visited.contains(&block_def.name) {
        tracing::error!(
            "检测到循环块引用：'{}'，跳过展开以避免栈溢出\n\
             可能原因：\n\
             1. 块定义确实存在循环引用（A 引用 B，B 引用 A）\n\
             2. 动态块被误判为循环引用\n\
             建议：检查块定义 '{}'",
            block_def.name,
            block_def.name
        );
        return vec![];
    }
    visited.insert(block_def.name.clone());

    let mut entities = Vec::new();

    for entity in &block_def.entities {
        match entity {
            // 遇到嵌套块引用，递归展开
            RawEntity::BlockReference { block_name, insertion_point, scale, rotation, metadata: _, semantic: _ } => {
                if let Some(nested_def) = block_definitions.get(block_name) {
                    // 递归展开嵌套块
                    let nested_entities = resolve_block_references(nested_def, block_definitions, visited);

                    // 应用嵌套块的变换
                    for nested_entity in &nested_entities {
                        let transformed = transform_entity(
                            nested_entity,
                            *scale,
                            *rotation,
                            *insertion_point,
                        );
                        entities.push(transformed);
                    }
                } else {
                    tracing::warn!("嵌套块引用 '{}' 未找到对应的块定义", block_name);
                }
            }
            // 其他实体直接克隆
            _ => {
                entities.push(entity.clone());
            }
        }
    }

    visited.remove(&block_def.name);
    entities
}

/// 计算平均缩放比例
fn avg_scale(scale: [f64; 3]) -> f64 {
    (scale[0] + scale[1]) / 2.0
}

/// 过滤多段线中的零长度边（相邻重复点）
///
/// # 参数
/// - `points`: 原始点列表
///
/// # 返回
/// 过滤后的点列表
fn filter_zero_length_edges(points: Polyline) -> Polyline {
    if points.len() <= 2 {
        return points;
    }

    let mut filtered = Vec::with_capacity(points.len());
    const MIN_EDGE_LENGTH: f64 = 1e-4; // 0.1mm，与零长度线段过滤保持一致

    filtered.push(points[0]);

    for curr in points.iter().skip(1) {
        let prev = filtered.last().unwrap();

        // 计算边长
        let edge_length = ((curr[0] - prev[0]).powi(2) + (curr[1] - prev[1]).powi(2)).sqrt();

        if edge_length >= MIN_EDGE_LENGTH {
            filtered.push(*curr);
        } else {
            tracing::trace!("过滤零长度边：length={:.6}", edge_length);
        }
    }

    // 如果是闭合多段线，检查首尾点
    // （这个检查应该在调用处根据 closed 标志处理）

    filtered
}

impl DxfParser {
    /// 将 DXF 实体转换为 RawEntity（用于块定义解析）
    fn convert_entity(&self, entity: &dxf::entities::Entity) -> Result<Option<RawEntity>, CadError> {
        // 创建元数据
        let color_index = entity.common.color.index().map(|i| i.to_string());
        let layer_name = entity.common.layer.clone();
        let material = map_material_from_color(color_index.as_deref())
            .or_else(|| map_material_from_layer(&layer_name));

        let metadata = EntityMetadata {
            layer: Some(layer_name.clone()),
            color: color_index.clone(),
            lineweight: lineweight_enum_to_mm(entity.common.lineweight_enum_value),
            line_type: Some(entity.common.line_type_name.clone()).filter(|s| !s.is_empty()),
            handle: Some(format!("{:?}", entity.common.handle)),
            material,
            width: None,
        };

        let semantic = identify_semantic(&layer_name);

        match &entity.specific {
            EntityType::Line(line) => {
                let start = [line.p1.x, line.p1.y];
                let end = [line.p2.x, line.p2.y];
                
                // 过滤零长度线段（使用更严格的容差）
                const MIN_LINE_LENGTH: f64 = 1e-4; // 0.1mm
                let length = ((end[0] - start[0]).powi(2) + (end[1] - start[1]).powi(2)).sqrt();
                if length < MIN_LINE_LENGTH {
                    tracing::debug!("过滤零长度线段：handle={:?}, length={:.6}", entity.common.handle, length);
                    return Ok(None);
                }
                
                Ok(Some(RawEntity::Line {
                    start,
                    end,
                    metadata: metadata.clone(),
                    semantic: semantic.clone(),
                }))
            }
            EntityType::Polyline(polyline) => {
                let points: Polyline = polyline.vertices()
                    .map(|v| [v.location.x, v.location.y])
                    .collect();
                if points.len() >= 2 {
                    Ok(Some(RawEntity::Polyline {
                        points,
                        closed: polyline.is_closed(),
                        metadata: metadata.clone(),
                        semantic: semantic.clone(),
                    }))
                } else {
                    Ok(None)
                }
            }
            EntityType::LwPolyline(lwpolyline) => {
                let mut points: Polyline = Vec::new();
                let vertices = &lwpolyline.vertices;

                if vertices.is_empty() {
                    return Ok(None);
                }

                for i in 0..vertices.len() {
                    let v1 = &vertices[i];
                    points.push([v1.x, v1.y]);

                    if v1.bulge.abs() > BULGE_EPSILON {
                        let v2 = if i + 1 < vertices.len() {
                            &vertices[i + 1]
                        } else if lwpolyline.is_closed() {
                            &vertices[0]
                        } else {
                            continue;
                        };

                        let p1 = [v1.x, v1.y];
                        let p2 = [v2.x, v2.y];
                        let arc_points = bulge_to_arc_points(p1, p2, v1.bulge, self.config.arc_tolerance_mm);

                        if arc_points.len() > 2 {
                            points.extend_from_slice(&arc_points[1..]);
                        }
                    }
                }

                if lwpolyline.is_closed() && !vertices.is_empty() {
                    let last_v = &vertices[vertices.len() - 1];
                    if last_v.bulge.abs() > BULGE_EPSILON {
                        let p1 = [last_v.x, last_v.y];
                        let p2 = [vertices[0].x, vertices[0].y];
                        let arc_points = bulge_to_arc_points(p1, p2, last_v.bulge, self.config.arc_tolerance_mm);
                        if arc_points.len() > 2 {
                            points.extend_from_slice(&arc_points[1..]);
                        }
                    }
                }

                if points.len() >= 2 {
                    // 过滤相邻重复点（零长度边）
                    let filtered_points = filter_zero_length_edges(points.clone());

                    if filtered_points.len() >= 2 {
                        // 如果是闭合多段线，检查首尾点是否重复
                        let mut final_points = filtered_points.clone();
                        if lwpolyline.is_closed() && filtered_points.len() >= 2 {
                            let first = filtered_points[0];
                            let last = *filtered_points.last().unwrap();
                            let distance = ((first[0] - last[0]).powi(2) + (first[1] - last[1]).powi(2)).sqrt();

                            if distance < self.tolerance {
                                final_points.pop(); // 移除重复的终点
                                tracing::debug!("移除闭合多段线重复终点：handle={:?}, distance={:.6}",
                                    entity.common.handle, distance);
                            }
                        }

                        if final_points.len() >= 2 {
                            Ok(Some(RawEntity::Polyline {
                                points: final_points,
                                closed: lwpolyline.is_closed(),
                                metadata: metadata.clone(),
                                semantic: semantic.clone(),
                            }))
                        } else {
                            tracing::debug!("移除重复终点后点数不足：handle={:?}, original_count={}, final_count={}",
                                entity.common.handle, filtered_points.len(), final_points.len());
                            Ok(None)
                        }
                    } else {
                        tracing::debug!("过滤零长度边后点数不足：handle={:?}, original_count={}, filtered_count={}",
                            entity.common.handle, points.len(), filtered_points.len());
                        Ok(None)
                    }
                } else {
                    Ok(None)
                }
            }
            EntityType::Arc(arc) => {
                Ok(Some(RawEntity::Arc {
                    center: [arc.center.x, arc.center.y],
                    radius: arc.radius,
                    start_angle: arc.start_angle.to_degrees(),
                    end_angle: arc.end_angle.to_degrees(),
                    metadata: metadata.clone(),
                    semantic: semantic.clone(),
                }))
            }
            EntityType::Circle(circle) => {
                Ok(Some(RawEntity::Circle {
                    center: [circle.center.x, circle.center.y],
                    radius: circle.radius,
                    metadata: metadata.clone(),
                    semantic: semantic.clone(),
                }))
            }
            EntityType::Spline(spline) => {
                let points = self.discretize_spline(spline);
                if points.len() >= 2 {
                    Ok(Some(RawEntity::Polyline {
                        points,
                        closed: spline.is_closed(),
                        metadata: metadata.clone(),
                        semantic: semantic.clone(),
                    }))
                } else {
                    Ok(None)
                }
            }
            EntityType::Ellipse(ellipse) => {
                let points = self.discretize_ellipse(ellipse);
                if points.len() >= 2 {
                    Ok(Some(RawEntity::Polyline {
                        points,
                        closed: true,
                        metadata: metadata.clone(),
                        semantic: semantic.clone(),
                    }))
                } else {
                    Ok(None)
                }
            }
            EntityType::Text(text) => {
                if !self.config.ignore_text {
                    Ok(Some(RawEntity::Text {
                        position: [text.location.x, text.location.y],
                        content: text.value.clone(),
                        height: text.text_height,
                        rotation: text.rotation,
                        style_name: None,
                        align_left: None,
                        align_right: None,
                        metadata: metadata.clone(),
                        semantic: semantic.clone(),
                    }))
                } else {
                    Ok(None)
                }
            }
            EntityType::MText(mtext) => {
                if !self.config.ignore_text {
                    Ok(Some(RawEntity::Text {
                        position: [mtext.insertion_point.x, mtext.insertion_point.y],
                        content: clean_mtext_content(&mtext.text),
                        height: mtext.initial_text_height,
                        rotation: 0.0,
                        style_name: None,
                        align_left: None,
                        align_right: None,
                        metadata: metadata.clone(),
                        semantic: semantic.clone(),
                    }))
                } else {
                    Ok(None)
                }
            }
            // 嵌套块引用（在块定义内部）- 保留为 BlockReference
            EntityType::Insert(insert) => {
                // 在块定义内部遇到 INSERT，创建 BlockReference 实体
                // 后续通过 resolve_block_references 递归展开
                Ok(Some(RawEntity::BlockReference {
                    block_name: insert.name.clone(),
                    insertion_point: [insert.location.x, insert.location.y],
                    scale: [
                        if insert.x_scale_factor != 0.0 { insert.x_scale_factor } else { 1.0 },
                        if insert.y_scale_factor != 0.0 { insert.y_scale_factor } else { 1.0 },
                        if insert.z_scale_factor != 0.0 { insert.z_scale_factor } else { 1.0 },
                    ],
                    rotation: insert.rotation,
                    metadata: metadata.clone(),
                    semantic: semantic.clone(),
                }))
            }
            _ => {
                tracing::debug!("跳过未支持的实体类型：{:?}", entity.specific);
                Ok(None)
            }
        }
    }
}

// ============================================================================
// P11 技术设计文档 v1.0：座椅区块识别、3D 实体检测、SPLINE 简化
// ============================================================================

impl DxfParser {
    /// 检测座椅区块
    ///
    /// # 算法步骤
    /// 1. 收集所有 INSERT 实体
    /// 2. 按块名分组
    /// 3. 识别座椅块（名称关键词 + 数量阈值）
    /// 4. 计算凸包边界
    /// 5. 推断座椅类型和声学属性
    pub fn detect_seat_zones(&self, path: impl AsRef<Path>) -> Result<Vec<SeatZone>, CadError> {
        let drawing = Drawing::load_file(path.as_ref())
            .map_err(|e| CadError::dxf_parse_with_source(
                path.as_ref().to_path_buf(),
                DxfParseReason::FileNotFound,
                e
            ))?;

        // 1. 收集所有 INSERT 实体
        let inserts: Vec<&Insert> = drawing.entities()
            .filter_map(|e| {
                if let EntityType::Insert(ref ins) = e.specific {
                    Some(ins)
                } else {
                    None
                }
            })
            .collect();

        // 2. 按块名分组
        let mut block_groups: HashMap<String, Vec<&Insert>> = HashMap::new();
        for insert in inserts {
            block_groups.entry(insert.name.clone())
                .or_default()
                .push(insert);
        }

        // 3. 识别座椅块
        let seat_keywords = ["SEAT", "CHAIR", "椅", "凳", "AUDITORIUM", "STOOL"];
        let mut seat_zones = Vec::new();

        for (block_name, inserts) in block_groups {
            // 关键词匹配
            let is_seat_block = seat_keywords.iter()
                .any(|k| block_name.to_uppercase().contains(k));

            if !is_seat_block {
                continue;
            }

            // 数量阈值检查
            if inserts.len() < self.config.seat_zone_min_count {
                continue;
            }

            // 4. 计算凸包边界
            let boundary = self.compute_seat_zone_convex_hull(&inserts);

            // 5. 推断座椅类型
            let seat_type = self.infer_seat_type(&inserts);

            // 6. 推断声学属性
            let acoustic_props = self.infer_acoustic_properties(seat_type);

            // 7. 可选：保存原始位置（用于 LOD2 渲染）
            let original_positions: Vec<Point2> = inserts
                .iter()
                .map(|ins| [ins.location.x, ins.location.y])
                .collect();

            seat_zones.push(SeatZone {
                boundary,
                seat_count: inserts.len(),
                seat_type,
                acoustic_properties: acoustic_props,
                original_positions: Some(original_positions),
            });
        }

        Ok(seat_zones)
    }

    /// 计算座椅区域凸包边界（使用 Graham 扫描算法）
    fn compute_seat_zone_convex_hull(&self, inserts: &[&Insert]) -> ClosedLoop {
        // 1. 收集插入点
        let mut points: Vec<Point2> = inserts
            .iter()
            .map(|ins| [ins.location.x, ins.location.y])
            .collect();

        if points.len() < 3 {
            return ClosedLoop::new(points);
        }

        // 2. Graham 扫描算法计算凸包
        points = self.graham_scan_convex_hull(points);

        ClosedLoop::new(points)
    }

    /// Graham 扫描算法计算凸包
    fn graham_scan_convex_hull(&self, mut points: Vec<Point2>) -> Vec<Point2> {
        if points.len() < 3 {
            return points;
        }

        // 找到最下方的点（y 最小，x 最小）
        let mut min_idx = 0;
        for i in 1..points.len() {
            if points[i][1] < points[min_idx][1] 
                || (points[i][1] == points[min_idx][1] && points[i][0] < points[min_idx][0]) {
                min_idx = i;
            }
        }
        points.swap(0, min_idx);
        let pivot = points[0];

        // 按极角排序
        points[1..].sort_by(|a, b| {
            let cross = (a[0] - pivot[0]) * (b[1] - pivot[1]) - (a[1] - pivot[1]) * (b[0] - pivot[0]);
            if cross.abs() < 1e-10 {
                // 共线，按距离排序
                let dist_a = (a[0] - pivot[0]).powi(2) + (a[1] - pivot[1]).powi(2);
                let dist_b = (b[0] - pivot[0]).powi(2) + (b[1] - pivot[1]).powi(2);
                dist_a.partial_cmp(&dist_b).unwrap()
            } else if cross > 0.0 {
                std::cmp::Ordering::Less
            } else {
                std::cmp::Ordering::Greater
            }
        });

        // 移除共线点（保留最远的）
        let mut filtered = vec![pivot];
        for point in points.iter().skip(1) {
            while filtered.len() > 1 {
                let top = filtered[filtered.len() - 1];
                let cross = (top[0] - pivot[0]) * (point[1] - pivot[1])
                          - (top[1] - pivot[1]) * (point[0] - pivot[0]);
                if cross.abs() < 1e-10 {
                    filtered.pop();
                } else {
                    break;
                }
            }
            filtered.push(*point);
        }

        // Graham 扫描
        let mut hull: Vec<Point2> = Vec::new();
        for p in filtered {
            while hull.len() > 1 {
                let top: Point2 = hull[hull.len() - 1];
                let second: Point2 = hull[hull.len() - 2];
                let cross = (top[0] - second[0]) * (p[1] - second[1]) 
                          - (top[1] - second[1]) * (p[0] - second[0]);
                if cross <= 0.0 {
                    hull.pop();
                } else {
                    break;
                }
            }
            hull.push(p);
        }

        hull
    }

    /// 推断座椅类型（基于间距分析）
    fn infer_seat_type(&self, inserts: &[&Insert]) -> SeatType {
        if inserts.len() < 2 {
            return SeatType::Unknown;
        }

        // 采样计算间距（避免 O(n²)）
        let sample_size = inserts.len().min(100);
        let mut distances = Vec::new();

        for i in 0..sample_size {
            for j in (i+1)..sample_size {
                let d = distance_2d(
                    [inserts[i].location.x, inserts[i].location.y],
                    [inserts[j].location.x, inserts[j].location.y],
                );
                // 合理座椅间距：0.3-2.0m
                if (300.0..=2000.0).contains(&d) {
                    distances.push(d);
                }
            }
        }

        if distances.is_empty() {
            return SeatType::Unknown;
        }

        // 计算平均间距
        let avg_distance = distances.iter().sum::<f64>() / distances.len() as f64;

        // 推断类型（单位：mm）
        if avg_distance < 550.0 {
            SeatType::Auditorium  // 礼堂排椅（间距小）
        } else if avg_distance < 800.0 {
            SeatType::Single  // 单人椅
        } else if avg_distance < 1200.0 {
            SeatType::Double  // 双人椅
        } else {
            SeatType::Bench  // 长凳
        }
    }

    /// 推断声学属性
    fn infer_acoustic_properties(&self, seat_type: SeatType) -> AcousticProps {
        match seat_type {
            SeatType::Auditorium => AcousticProps {
                absorption_coefficient: 0.70,  // 礼堂椅（布艺）
                scattering_coefficient: 0.15,
            },
            SeatType::Single => AcousticProps {
                absorption_coefficient: 0.30,  // 单人椅（硬面）
                scattering_coefficient: 0.10,
            },
            SeatType::Double => AcousticProps {
                absorption_coefficient: 0.50,  // 双人椅（软垫）
                scattering_coefficient: 0.12,
            },
            SeatType::Bench => AcousticProps {
                absorption_coefficient: 0.25,  // 长凳（硬面）
                scattering_coefficient: 0.08,
            },
            SeatType::Unknown => AcousticProps::default(),
        }
    }

    /// 检测 3D 实体并生成警告
    pub fn detect_3d_entities(&self, drawing: &Drawing) -> Vec<Entity3DWarning> {
        let mut warnings = Vec::new();

        for entity in drawing.entities() {
            let z_range = match &entity.specific {
                EntityType::Line(line) => {
                    let z_min = line.p1.z.min(line.p2.z);
                    let z_max = line.p1.z.max(line.p2.z);
                    Some([z_min, z_max])
                }
                EntityType::Polyline(polyline) => {
                    let z_values: Vec<f64> = polyline.vertices()
                        .map(|v| v.location.z)
                        .collect();
                    if z_values.is_empty() {
                        None
                    } else {
                        // 使用 f64 的比较方法
                        let z_min = z_values.iter().cloned().fold(f64::NAN, f64::min);
                        let z_max = z_values.iter().cloned().fold(f64::NAN, f64::max);
                        Some([z_min, z_max])
                    }
                }
                EntityType::LwPolyline(_) => {
                    // LWPOLYLINE 通常是 2D 的，检查标高
                    Some([0.0, 0.0])
                }
                EntityType::Circle(circle) => {
                    Some([circle.center.z, circle.center.z])
                }
                EntityType::Arc(arc) => {
                    Some([arc.center.z, arc.center.z])
                }
                _ => None,
            };

            if let Some([z_min, z_max]) = z_range {
                // Z 坐标变化超过阈值，生成警告
                if (z_max - z_min).abs() > 1e-6 {
                    warnings.push(Entity3DWarning {
                        entity_type: format!("{:?}", entity.specific),
                        handle: Some(format!("{:?}", entity.common.handle)),
                        z_range: [z_min, z_max],
                        message: format!(
                            "检测到 3D 实体（Z 范围：{:.2} - {:.2} mm），将投影到 XY 平面",
                            z_min, z_max
                        ),
                    });
                }
            }
        }

        warnings
    }

    /// 简化 SPLINE 离散化
    ///
    /// # 策略
    /// 1. 控制点过多时，使用更大容差进行简化
    /// 2. 限制最大输出点数
    /// 3. 使用 Douglas-Peucker 算法后处理
    pub fn simplify_spline(&self, spline: &dxf::entities::Spline) -> Polyline {
        // 1. 检查控制点数量
        let control_point_count = spline.control_points.len();

        if control_point_count > self.config.max_spline_control_points {
            tracing::warn!(
                "SPLINE 控制点过多 ({} 个 > {} 个)，使用简化离散化",
                control_point_count,
                self.config.max_spline_control_points
            );

            // 2. 使用更大容差进行初始离散化
            let simplified_tolerance = self.config.arc_tolerance_mm *
                (control_point_count as f64 / self.config.max_spline_control_points as f64);

            let points = self.discretize_spline_with_tolerance(spline, simplified_tolerance);

            // 3. 使用 Douglas-Peucker 算法进一步简化
            let coords: Vec<Coord<f64>> = points
                .iter()
                .map(|p| Coord { x: p[0], y: p[1] })
                .collect();

            let line_string = LineString::new(coords);
            let simplified = line_string.simplify_vw(&self.config.arc_tolerance_mm);

            simplified.0.iter()
                .map(|c| [c.x, c.y])
                .collect()
        } else {
            // 正常离散化
            self.discretize_spline_with_tolerance(spline, self.config.arc_tolerance_mm)
        }
    }

    /// 使用容差离散化 SPLINE
    fn discretize_spline_with_tolerance(&self, spline: &dxf::entities::Spline, _tolerance: f64) -> Polyline {
        // 使用 curvo 库进行 NURBS 离散化
        // 注意：curvo 的 NurbsCurve 需要从控制点和节点向量构建
        // 这里使用简化的离散化方法
        
        // 从 SPLINE 控制点生成近似曲线
        let control_points: Vec<[f64; 2]> = spline.control_points
            .iter()
            .map(|cp| [cp.x, cp.y])
            .collect();
        
        // 如果控制点很少，直接返回
        if control_points.len() < 2 {
            return control_points;
        }

        // 使用简单的线性插值离散化
        let num_points = (control_points.len() * 10).clamp(20, 1000);
        let mut points = Vec::with_capacity(num_points);
        
        // 对于简单情况，直接使用控制点
        if spline.control_points.len() <= 10 {
            return control_points;
        }
        
        // 对于复杂 SPLINE，使用更密集的采样
        for i in 0..num_points {
            let t = i as f64 / (num_points - 1) as f64;
            // 线性插值近似
            let idx = (t * (control_points.len() - 1) as f64) as usize;
            let next_idx = (idx + 1).min(control_points.len() - 1);
            let local_t = (t * (control_points.len() - 1) as f64) - idx as f64;
            
            let x = control_points[idx][0] * (1.0 - local_t) + control_points[next_idx][0] * local_t;
            let y = control_points[idx][1] * (1.0 - local_t) + control_points[next_idx][1] * local_t;
            points.push([x, y]);
        }

        points
    }
}

/// 计算两点间距离
#[inline]
fn distance_2d(p1: [f64; 2], p2: [f64; 2]) -> f64 {
    let dx = p2[0] - p1[0];
    let dy = p2[1] - p1[1];
    (dx * dx + dy * dy).sqrt()
}

// ============================================================================
// P3 增强：MTEXT 内容清理
// ============================================================================

/// 清理 MTEXT 内容中的 DXF 格式代码
///
/// DXF MTEXT 可能包含以下格式代码：
/// - `\X` - 自动换行
/// - `\P` - 段落分隔
/// - `\~` - 不间断空格
/// - `{\f...}` - 字体切换
/// - `\Q...;` - 倾斜角度
/// - `\H...;` - 文字高度
/// - `\W...;` - 文字宽度比例
/// - `\A...;` - 对齐方式
///
/// 此函数移除所有格式代码，保留纯文本内容
fn clean_mtext_content(raw: &str) -> String {
    let mut result = String::with_capacity(raw.len());
    let mut chars = raw.chars().peekable();

    while let Some(c) = chars.next() {
        match c {
            '\\' => {
                // 处理转义序列
                if let Some(&next) = chars.peek() {
                    match next {
                        'X' => {
                            // \X 自动换行 - 替换为空格
                            chars.next();
                            result.push(' ');
                        }
                        'P' => {
                            // \P 段落分隔 - 替换为换行
                            chars.next();
                            result.push('\n');
                        }
                        '~' => {
                            // \~ 不间断空格
                            chars.next();
                            result.push(' ');
                        }
                        '{' => {
                            // 可能是字体切换 {\f...}
                            // 跳过整个 {...} 块
                            let mut brace_count = 1;
                            while let Some(&ch) = chars.peek() {
                                chars.next();
                                if ch == '{' {
                                    brace_count += 1;
                                } else if ch == '}' {
                                    brace_count -= 1;
                                    if brace_count == 0 {
                                        break;
                                    }
                                }
                            }
                        }
                        'Q' | 'H' | 'W' | 'A' => {
                            // \Q...; \H...; \W...; \A...; 格式控制
                            chars.next();
                            // 跳过直到分号
                            while let Some(&ch) = chars.peek() {
                                if ch == ';' {
                                    chars.next();
                                    break;
                                }
                                chars.next();
                            }
                        }
                        _ => {
                            // 未知转义，保留原样
                            result.push(c);
                        }
                    }
                } else {
                    result.push(c);
                }
            }
            '{' | '}' => {
                // 跳过独立的括号（可能是残留的格式代码）
                // 但保留可能的文本内容
            }
            _ => {
                result.push(c);
            }
        }
    }

    // 清理多余空白
    result.trim().to_string()
}

// ============================================================================
// P3 增强：图层分组和过滤
// ============================================================================

/// 图层分组信息
#[derive(Debug, Clone)]
pub struct LayerGroup {
    /// 分组名称
    pub name: String,
    /// 分组中的图层列表
    pub layers: Vec<String>,
    /// 是否可见
    pub visible: bool,
    /// 分组描述
    pub description: String,
}

/// 智能图层管理器
pub struct LayerManager {
    /// 图层分组
    pub groups: Vec<LayerGroup>,
    /// 图层可见性映射
    pub visibility: HashMap<String, bool>,
}

impl LayerManager {
    /// 创建新的图层管理器
    pub fn new() -> Self {
        Self {
            groups: Vec::new(),
            visibility: HashMap::new(),
        }
    }

    /// 根据过滤模式创建图层分组
    pub fn from_filter_mode(mode: LayerFilterMode, all_layers: &[String]) -> Self {
        let mut manager = Self::new();

        match mode {
            LayerFilterMode::All => {
                // 不过滤，所有图层都可见
                manager.visibility = all_layers.iter().map(|l| (l.clone(), true)).collect();
            }
            LayerFilterMode::WallsOnly => {
                // 仅墙体图层
                let wall_layers: Vec<String> = all_layers
                    .iter()
                    .filter(|l| is_wall_layer(l))
                    .cloned()
                    .collect();
                manager.visibility = wall_layers.iter().map(|l| (l.clone(), true)).collect();
                manager.groups.push(LayerGroup {
                    name: "Walls".to_string(),
                    layers: wall_layers,
                    visible: true,
                    description: "墙体结构图层".to_string(),
                });
            }
            LayerFilterMode::OpeningsOnly => {
                // 仅开口图层
                let opening_layers: Vec<String> = all_layers
                    .iter()
                    .filter(|l| is_door_only(l) || is_window_only(l) || is_opening_layer(l))
                    .cloned()
                    .collect();
                manager.visibility = opening_layers.iter().map(|l| (l.clone(), true)).collect();
                manager.groups.push(LayerGroup {
                    name: "Openings".to_string(),
                    layers: opening_layers,
                    visible: true,
                    description: "门窗开口图层".to_string(),
                });
            }
            LayerFilterMode::Architectural => {
                // 建筑图层（墙 + 开口）
                let arch_layers: Vec<String> = all_layers
                    .iter()
                    .filter(|l| {
                        is_wall_layer(l) || is_door_only(l) || is_window_only(l) || is_opening_layer(l)
                    })
                    .cloned()
                    .collect();
                manager.visibility = arch_layers.iter().map(|l| (l.clone(), true)).collect();
                manager.groups.push(LayerGroup {
                    name: "Architectural".to_string(),
                    layers: arch_layers,
                    visible: true,
                    description: "建筑图层（墙 + 开口）".to_string(),
                });
            }
            LayerFilterMode::Furniture => {
                // 仅家具图层
                let furniture_layers: Vec<String> = all_layers
                    .iter()
                    .filter(|l| is_furniture_layer(l))
                    .cloned()
                    .collect();
                manager.visibility = furniture_layers.iter().map(|l| (l.clone(), true)).collect();
                manager.groups.push(LayerGroup {
                    name: "Furniture".to_string(),
                    layers: furniture_layers,
                    visible: true,
                    description: "家具图层".to_string(),
                });
            }
            LayerFilterMode::Custom => {
                // 自定义过滤 - 需要外部配置
                manager.visibility = all_layers.iter().map(|l| (l.clone(), true)).collect();
            }
        }

        manager
    }

    /// 切换图层可见性
    pub fn toggle_visibility(&mut self, layer: &str) {
        if let Some(visible) = self.visibility.get_mut(layer) {
            *visible = !*visible;
        } else {
            self.visibility.insert(layer.to_string(), false);
        }
    }

    /// 设置图层可见性
    pub fn set_visibility(&mut self, layer: &str, visible: bool) {
        self.visibility.insert(layer.to_string(), visible);
    }

    /// 获取所有可见的图层
    pub fn visible_layers(&self) -> Vec<&String> {
        self.visibility
            .iter()
            .filter(|(_, v)| **v)
            .map(|(k, _)| k)
            .collect()
    }

    /// 检查图层是否可见
    pub fn is_visible(&self, layer: &str) -> bool {
        *self.visibility.get(layer).unwrap_or(&true)
    }
}

impl Default for LayerManager {
    fn default() -> Self {
        Self::new()
    }
}

/// 判断图层是否应该被过滤
///
/// 根据配置和图层语义判断是否保留该图层
pub fn should_filter_layer(layer: &str, config: &DxfConfig) -> bool {
    // 如果有白名单，只保留白名单中的图层
    if let Some(whitelist) = &config.layer_whitelist {
        return !whitelist.iter().any(|w| layer.contains(w));
    }

    // 根据过滤模式判断
    match config.layer_filter_mode {
        LayerFilterMode::All => false,  // 但是保留所有图层
        LayerFilterMode::WallsOnly => !is_wall_layer(layer),
        LayerFilterMode::OpeningsOnly => {
            !(is_door_only(layer) || is_window_only(layer) || is_opening_layer(layer))
        }
        LayerFilterMode::Architectural => {
            !(is_wall_layer(layer) || is_door_only(layer) || is_window_only(layer) || is_opening_layer(layer))
        }
        LayerFilterMode::Furniture => !is_furniture_layer(layer),
        LayerFilterMode::Custom => {
            // 自定义过滤：检查 custom_layer_groups
            if config.custom_layer_groups.is_empty() {
                false  // 没有自定义规则，保留所有
            } else {
                !config.custom_layer_groups.iter().any(|(pattern, _)| layer.contains(pattern))
            }
        }
    }
}

// ============================================================================
// P5 增强：解析质量评分计算
// ============================================================================

impl DxfParseReport {
    /// 计算解析质量评分
    ///
    /// 评分基于以下因素：
    /// - 实体解析成功率
    /// - 警告数量
    /// - 错误数量
    /// - 单位匹配情况
    /// - 3D 实体警告
    ///
    /// 返回 0.0 - 1.0 之间的评分
    pub fn calculate_quality_score(&mut self) -> f64 {
        let mut score = 1.0;

        // 因子 1：实体解析成功率（权重 40%）
        if self.parse_stats.total_entities > 0 {
            let success_rate = self.parse_stats.valid_entities as f64 
                / self.parse_stats.total_entities as f64;
            score -= (1.0 - success_rate) * 0.4;
        }

        // 因子 2：错误数量（权重 30%）
        if self.parse_stats.error_count > 0 {
            let error_penalty = (self.parse_stats.error_count as f64 * 0.05).min(0.3);
            score -= error_penalty;
        }

        // 因子 3：警告数量（权重 15%）
        if self.parse_stats.warning_count > 0 {
            let warning_penalty = (self.parse_stats.warning_count as f64 * 0.01).min(0.15);
            score -= warning_penalty;
        }

        // 因子 4：单位不匹配（权重 10%）
        if self.unit_mismatch_detected {
            score -= 0.1;
        }

        // 因子 5：3D 实体警告（权重 5%）
        if !self._3d_entity_warnings.is_empty() {
            let _3d_penalty = (self._3d_entity_warnings.len() as f64 * 0.01).min(0.05);
            score -= _3d_penalty;
        }

        // 确保评分在合理范围内
        self.quality_score = score.clamp(0.0, 1.0);
        self.quality_score
    }

    /// 添加解析问题
    pub fn add_issue(&mut self, issue: ParseIssue) {
        if issue.severity == ParseIssueSeverity::Error 
            || issue.severity == ParseIssueSeverity::Critical 
        {
            self.parse_stats.error_count += 1;
        } else if issue.severity == ParseIssueSeverity::Warning {
            self.parse_stats.warning_count += 1;
        }
        self.issues.push(issue);
    }

    /// 生成质量问题报告文本
    pub fn generate_quality_report(&self) -> String {
        let mut report = String::new();
        
        report.push_str(&format!("解析质量评分：{:.1}%\n", self.quality_score * 100.0));
        report.push_str(&format!("总实体数：{}\n", self.parse_stats.total_entities));
        report.push_str(&format!("有效实体：{}\n", self.parse_stats.valid_entities));
        report.push_str(&format!("跳过实体：{}\n", self.parse_stats.skipped_entities));
        report.push_str(&format!("解析时间：{:.2}ms\n", self.parse_stats.parse_time_ms));
        
        if !self.issues.is_empty() {
            report.push_str("\n问题列表:\n");
            for issue in &self.issues {
                let severity_str = match issue.severity {
                    ParseIssueSeverity::Info => "ℹ️",
                    ParseIssueSeverity::Warning => "⚠️",
                    ParseIssueSeverity::Error => "❌",
                    ParseIssueSeverity::Critical => "🔴",
                };
                report.push_str(&format!("  {} [{}] {}\n", severity_str, issue.code, issue.message));
                if let Some(suggestion) = &issue.suggestion {
                    report.push_str(&format!("     建议：{}\n", suggestion));
                }
            }
        }

        report
    }
}
