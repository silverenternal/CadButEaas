//! 几何基础类型定义

use serde::{Deserialize, Serialize};
use schemars::JsonSchema;
use crate::scene::{BoundarySemantic, LengthUnit};

/// 2D 点类型 (单位：mm)
pub type Point2 = [f64; 2];

/// 3D 点类型 (单位：mm)
pub type Point3 = [f64; 3];

/// 多段线 (polyline)
pub type Polyline = Vec<Point2>;

/// RGB 颜色（8 位每通道）
#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema)]
pub struct Color32 {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: u8,
}

impl Color32 {
    pub const fn new(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self { r, g, b, a }
    }

    pub const BLACK: Self = Self::new(0, 0, 0, 255);
    pub const WHITE: Self = Self::new(255, 255, 255, 255);
    pub const RED: Self = Self::new(255, 0, 0, 255);
    pub const GREEN: Self = Self::new(0, 255, 0, 255);
    pub const BLUE: Self = Self::new(0, 0, 255, 255);

    /// 转换为归一化的 RGBA 颜色（0.0-1.0）
    pub fn to_normalized(&self) -> [f32; 4] {
        [
            self.r as f32 / 255.0,
            self.g as f32 / 255.0,
            self.b as f32 / 255.0,
            self.a as f32 / 255.0,
        ]
    }
}

impl Default for Color32 {
    fn default() -> Self {
        Self::BLACK
    }
}

// ============================================================================
// P0-4: 线型支持
// ============================================================================

/// 线型（用于 CAD 制图中的虚线、点划线等）
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum LineStyle {
    /// 实线
    #[default]
    Solid,
    /// 虚线（短划线）
    Dashed,
    /// 点线
    Dotted,
    /// 点划线
    DashDot,
    /// 双点划线
    DashDotDot,
    /// 长划线
    LongDash,
    /// 长划 - 点 - 长划
    LongDashDot,
    /// 自定义线型（自定义模式）
    Custom,
}

impl LineStyle {
    /// 获取线型的模式（用于 GPU 渲染）
    /// 返回 [短划线长度，间隔长度，...] 的数组
    pub fn pattern(&self) -> &'static [f32] {
        match self {
            LineStyle::Solid => &[],
            LineStyle::Dashed => &[12.0, 6.0],        // 12px 划线，6px 间隔
            LineStyle::Dotted => &[2.0, 6.0],         // 2px 点，6px 间隔
            LineStyle::DashDot => &[12.0, 6.0, 2.0, 6.0],  // 12px 划线，6px 间隔，2px 点，6px 间隔
            LineStyle::DashDotDot => &[12.0, 6.0, 2.0, 6.0, 2.0, 6.0],
            LineStyle::LongDash => &[24.0, 6.0],      // 24px 长划线，6px 间隔
            LineStyle::LongDashDot => &[24.0, 6.0, 2.0, 6.0],
            LineStyle::Custom => &[10.0, 5.0],        // 默认自定义模式
        }
    }

    /// 从 DXF 线型名称转换
    pub fn from_dxf_name(name: &str) -> Option<Self> {
        match name.to_uppercase().as_str() {
            "CONTINUOUS" | "SOLID" => Some(LineStyle::Solid),
            "DASHED" | "ACAD_ISO02W100" => Some(LineStyle::Dashed),
            "DOT" | "ACAD_ISO03W100" => Some(LineStyle::Dotted),
            "DASHDOT" | "ACAD_ISO04W100" => Some(LineStyle::DashDot),
            "DIVIDE" | "ACAD_ISO05W100" => Some(LineStyle::DashDotDot),
            "CENTER" | "ACAD_ISO06W100" => Some(LineStyle::LongDashDot),
            _ => None,
        }
    }
}

/// 线宽（24 级，符合 CAD 标准）
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum LineWidth {
    /// 0.00mm (最细)
    W0,
    /// 0.05mm
    W1,
    /// 0.09mm
    W2,
    /// 0.13mm
    W3,
    /// 0.15mm
    W4,
    /// 0.18mm
    W5,
    /// 0.20mm
    W6,
    /// 0.25mm
    W7,
    /// 0.30mm
    W8,
    /// 0.35mm
    W9,
    /// 0.40mm
    W10,
    /// 0.50mm
    W11,
    /// 0.53mm
    W12,
    /// 0.60mm
    W13,
    /// 0.70mm
    W14,
    /// 0.80mm
    W15,
    /// 0.90mm
    W16,
    /// 1.00mm
    W17,
    /// 1.06mm
    W18,
    /// 1.20mm
    W19,
    /// 1.40mm
    W20,
    /// 1.58mm
    W21,
    /// 2.00mm
    W22,
    /// 2.11mm
    W23,
    /// ByLayer (跟随图层)
    #[default]
    ByLayer,
}

impl LineWidth {
    /// 获取线宽的毫米值
    pub fn to_mm(&self) -> Option<f32> {
        match self {
            LineWidth::W0 => Some(0.00),
            LineWidth::W1 => Some(0.05),
            LineWidth::W2 => Some(0.09),
            LineWidth::W3 => Some(0.13),
            LineWidth::W4 => Some(0.15),
            LineWidth::W5 => Some(0.18),
            LineWidth::W6 => Some(0.20),
            LineWidth::W7 => Some(0.25),
            LineWidth::W8 => Some(0.30),
            LineWidth::W9 => Some(0.35),
            LineWidth::W10 => Some(0.40),
            LineWidth::W11 => Some(0.50),
            LineWidth::W12 => Some(0.53),
            LineWidth::W13 => Some(0.60),
            LineWidth::W14 => Some(0.70),
            LineWidth::W15 => Some(0.80),
            LineWidth::W16 => Some(0.90),
            LineWidth::W17 => Some(1.00),
            LineWidth::W18 => Some(1.06),
            LineWidth::W19 => Some(1.20),
            LineWidth::W20 => Some(1.40),
            LineWidth::W21 => Some(1.58),
            LineWidth::W22 => Some(2.00),
            LineWidth::W23 => Some(2.11),
            LineWidth::ByLayer => None,
        }
    }

    /// 从毫米值创建线宽（四舍五入到最近的级别）
    pub fn from_mm(mm: f32) -> Self {
        let levels = [
            (0.00, LineWidth::W0),
            (0.05, LineWidth::W1),
            (0.09, LineWidth::W2),
            (0.13, LineWidth::W3),
            (0.15, LineWidth::W4),
            (0.18, LineWidth::W5),
            (0.20, LineWidth::W6),
            (0.25, LineWidth::W7),
            (0.30, LineWidth::W8),
            (0.35, LineWidth::W9),
            (0.40, LineWidth::W10),
            (0.50, LineWidth::W11),
            (0.53, LineWidth::W12),
            (0.60, LineWidth::W13),
            (0.70, LineWidth::W14),
            (0.80, LineWidth::W15),
            (0.90, LineWidth::W16),
            (1.00, LineWidth::W17),
            (1.06, LineWidth::W18),
            (1.20, LineWidth::W19),
            (1.40, LineWidth::W20),
            (1.58, LineWidth::W21),
            (2.00, LineWidth::W22),
            (2.11, LineWidth::W23),
        ];

        levels
            .iter()
            .min_by(|(a, _), (b, _)| {
                let diff_a = (a - mm).abs();
                let diff_b = (b - mm).abs();
                diff_a.partial_cmp(&diff_b).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(_, lw)| *lw)
            .unwrap_or(LineWidth::W7)
    }

    /// 获取像素宽度（用于渲染）
    pub fn to_pixels(&self, dpi: f32) -> f32 {
        match self.to_mm() {
            Some(mm) => mm * dpi / 25.4,  // mm 转像素
            None => 1.0,  // ByLayer 使用默认宽度
        }
    }
}


/// 几何容差配置 (单位：mm)
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ToleranceConfig {
    /// 端点吸附容差
    pub snap_tolerance: f64,
    /// 最小线段长度
    pub min_line_length: f64,
    /// 最大角度偏差 (度)
    pub max_angle_deviation: f64,
    /// 图纸单位（可选，用于单位转换）
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub units: Option<LengthUnit>,
}

impl Default for ToleranceConfig {
    fn default() -> Self {
        Self {
            snap_tolerance: 0.5,
            min_line_length: 1.0,
            max_angle_deviation: 5.0,
            units: Some(LengthUnit::Mm),
        }
    }
}

/// 标准化几何原语
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum RawEntity {
    Line {
        start: Point2,
        end: Point2,
        metadata: EntityMetadata,
        semantic: Option<BoundarySemantic>,
    },
    Polyline {
        points: Polyline,
        closed: bool,
        metadata: EntityMetadata,
        semantic: Option<BoundarySemantic>,
    },
    Arc {
        center: Point2,
        radius: f64,
        start_angle: f64,
        end_angle: f64,
        metadata: EntityMetadata,
        semantic: Option<BoundarySemantic>,
    },
    Circle {
        center: Point2,
        radius: f64,
        metadata: EntityMetadata,
        semantic: Option<BoundarySemantic>,
    },
    Text {
        position: Point2,
        content: String,
        height: f64,
        rotation: f64,           // 旋转角度（度）
        style_name: Option<String>,  // 文字样式名
        align_left: Option<Point2>,  // 左对齐点（用于对齐文字）
        align_right: Option<Point2>, // 右对齐点（用于对齐文字）
        metadata: EntityMetadata,
        semantic: Option<BoundarySemantic>,
    },
    Path {
        commands: Vec<PathCommand>,
        metadata: EntityMetadata,
        semantic: Option<BoundarySemantic>,
    },
    /// 块引用（INSERT 实体）
    BlockReference {
        block_name: String,
        insertion_point: Point2,
        scale: [f64; 3],  // X, Y, Z 缩放
        rotation: f64,    // 旋转角度（度）
        metadata: EntityMetadata,
        semantic: Option<BoundarySemantic>,
    },
    /// 尺寸标注
    Dimension {
        dimension_type: DimensionType,
        measurement: f64,
        text: Option<String>,
        definition_points: Vec<Point2>,
        metadata: EntityMetadata,
        semantic: Option<BoundarySemantic>,
    },
    /// 填充图案（HATCH 实体）- P0-1 建筑 CAD 核心支持
    Hatch {
        boundary_paths: Vec<HatchBoundaryPath>,
        pattern: HatchPattern,
        solid_fill: bool,
        metadata: EntityMetadata,
        semantic: Option<BoundarySemantic>,
    },
    /// P1-1: 外部参照（XREF）- 指向另一个 DWG/DXF 文件的引用
    XRef {
        /// 外部文件路径
        file_path: String,
        /// 插入点
        insertion_point: Point2,
        /// 缩放比例 [X, Y, Z]
        scale: [f64; 3],
        /// 旋转角度（度）
        rotation: f64,
        /// 参照类型（附着型/覆盖型）
        xref_type: XRefType,
        /// 图层名称（可选，用于过滤）
        layer_name: Option<String>,
        metadata: EntityMetadata,
        semantic: Option<BoundarySemantic>,
    },
}

/// P1-1: 外部参照类型
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum XRefType {
    /// 附着型（Attachment）- 递归加载外部参照
    #[default]
    Attachment,
    /// 覆盖型（Overlay）- 不递归加载
    Overlay,
}

/// 尺寸标注类型
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum DimensionType {
    Linear,      // 线性标注
    Aligned,     // 对齐标注
    Angular,     // 角度标注
    Radial,      // 半径标注
    Diameter,    // 直径标注
    ArcLength,   // 弧长标注（P0-2 新增）
    Ordinate,    // 坐标标注（P0-2 新增）
}

// ============================================================================
// HATCH 填充图案支持（P0-1：建筑 CAD 核心功能）
// ============================================================================

/// 填充图案类型
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum HatchPattern {
    /// 预定义图案（ANSI/ISO/建筑专用）
    Predefined {
        name: String,  // "ANSI31", "ANSI37", "AR-BRSTD", "AR-CONC" 等
    },
    /// 自定义图案（用户定义）
    Custom {
        pattern_def: HatchPatternDefinition,
    },
    /// 纯色填充
    Solid {
        color: Color32,
    },
}

/// 填充图案定义（自定义图案）
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct HatchPatternDefinition {
    /// 图案名称
    pub name: String,
    /// 图案描述
    pub description: Option<String>,
    /// 图案行定义
    pub lines: Vec<HatchPatternLine>,
}

/// 填充图案行定义
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct HatchPatternLine {
    /// 起始点 [x, y]
    pub start_point: Point2,
    /// 角度（度）
    pub angle: f64,
    /// 偏移量 [dx, dy]
    pub offset: Point2,
    /// 虚线模式（dash, gap, dash, gap, ...）
    pub dash_pattern: Vec<f64>,
}

/// 填充边界路径
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum HatchBoundaryPath {
    /// 多段线边界
    Polyline {
        points: Polyline,
        closed: bool,
    },
    /// 圆弧边界
    Arc {
        center: Point2,
        radius: f64,
        start_angle: f64,
        end_angle: f64,
        ccw: bool,  // 逆时针
    },
    /// 椭圆弧边界
    EllipseArc {
        center: Point2,
        major_axis: Point2,
        minor_axis_ratio: f64,
        start_angle: f64,
        end_angle: f64,
        ccw: bool,
    },
    /// 样条曲线边界
    Spline {
        control_points: Polyline,
        knots: Vec<f64>,
        degree: u32,
    },
}

/// 填充区域（离散化后的填充）
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct HatchRegion {
    /// 填充多边形（三角剖分结果）
    pub polygons: Vec<Vec<Point2>>,
    /// 填充图案
    pub pattern: HatchPattern,
    /// 填充比例
    pub scale: f64,
    /// 填充角度（度）
    pub angle: f64,
}

/// 块定义
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct BlockDefinition {
    pub name: String,
    pub base_point: Point2,
    pub entities: Vec<RawEntity>,
    pub metadata: EntityMetadata,
}

// ============================================================================
// 参数化块系统（P1-2）
// ============================================================================

/// 参数类型定义
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ParameterType {
    /// 长度参数（单位：mm）
    Length { default: f64, min: Option<f64>, max: Option<f64> },
    /// 角度参数（单位：度）
    Angle { default: f64, min: Option<f64>, max: Option<f64> },
    /// 布尔参数
    Boolean { default: bool },
    /// 枚举参数
    Enum { default: String, options: Vec<String> },
    /// 字符串参数
    String { default: String },
    /// 点数参数（用于阵列）
    Integer { default: i32, min: Option<i32>, max: Option<i32> },
}

impl ParameterType {
    /// 获取默认值的 JSON 表示
    pub fn default_value(&self) -> serde_json::Value {
        match self {
            ParameterType::Length { default, .. } => serde_json::json!(*default),
            ParameterType::Angle { default, .. } => serde_json::json!(*default),
            ParameterType::Boolean { default } => serde_json::json!(*default),
            ParameterType::Enum { default, .. } => serde_json::json!(*default),
            ParameterType::String { default } => serde_json::json!(*default),
            ParameterType::Integer { default, .. } => serde_json::json!(*default),
        }
    }
    
    /// 验证值是否有效
    pub fn validate(&self, value: &serde_json::Value) -> bool {
        match (self, value) {
            (ParameterType::Length { min, max, .. }, serde_json::Value::Number(v)) => {
                if let Some(v) = v.as_f64() {
                    min.map_or(true, |m| v >= m) && max.map_or(true, |m| v <= m)
                } else {
                    false
                }
            }
            (ParameterType::Angle { min, max, .. }, serde_json::Value::Number(v)) => {
                if let Some(v) = v.as_f64() {
                    min.map_or(true, |m| v >= m) && max.map_or(true, |m| v <= m)
                } else {
                    false
                }
            }
            (ParameterType::Boolean { .. }, serde_json::Value::Bool(_)) => true,
            (ParameterType::Enum { options, .. }, serde_json::Value::String(v)) => {
                options.contains(v)
            }
            (ParameterType::String { .. }, serde_json::Value::String(_)) => true,
            (ParameterType::Integer { min, max, .. }, serde_json::Value::Number(v)) => {
                if let Some(v) = v.as_i64() {
                    min.map_or(true, |m| v >= m as i64) && max.map_or(true, |m| v <= m as i64)
                } else {
                    false
                }
            }
            _ => false,
        }
    }
}

/// 参数定义
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ParameterDefinition {
    /// 参数名称
    pub name: String,
    /// 参数类型
    pub param_type: ParameterType,
    /// 参数描述
    pub description: Option<String>,
    /// 参数分组（用于 UI 组织）
    pub group: Option<String>,
    /// 是否可见（用于控制参数显示）
    pub visible: bool,
}

/// 参数约束
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ParameterConstraint {
    /// 等式约束：param1 = param2
    Equal { params: Vec<String> },
    /// 比例约束：param1 = scale * param2
    Ratio { param1: String, param2: String, scale: f64 },
    /// 公式约束：target = formula(other_params)
    Formula { target: String, formula: String },
    /// 范围约束：min <= param <= max
    Range { param: String, min: f64, max: f64 },
}

/// 参数化块定义
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ParametricBlockDefinition {
    /// 块名称
    pub name: String,
    /// 基础点
    pub base_point: Point2,
    /// 几何实体（支持参数引用）
    pub entities: Vec<RawEntity>,
    /// 参数定义
    pub parameters: Vec<ParameterDefinition>,
    /// 参数约束
    pub constraints: Vec<ParameterConstraint>,
    /// 元数据
    pub metadata: EntityMetadata,
    /// 版本（用于参数更新时的兼容性检查）
    pub version: u32,
}

impl ParametricBlockDefinition {
    /// 创建新的参数化块定义
    pub fn new(name: impl Into<String>, base_point: Point2) -> Self {
        Self {
            name: name.into(),
            base_point,
            entities: Vec::new(),
            parameters: Vec::new(),
            constraints: Vec::new(),
            metadata: EntityMetadata::default(),
            version: 1,
        }
    }
    
    /// 添加参数
    pub fn add_parameter(mut self, param: ParameterDefinition) -> Self {
        self.parameters.push(param);
        self
    }
    
    /// 添加几何实体
    pub fn add_entity(mut self, entity: RawEntity) -> Self {
        self.entities.push(entity);
        self
    }
    
    /// 添加约束
    pub fn add_constraint(mut self, constraint: ParameterConstraint) -> Self {
        self.constraints.push(constraint);
        self
    }
    
    /// 获取参数的默认值集合
    pub fn default_parameter_values(&self) -> serde_json::Map<String, serde_json::Value> {
        let mut values = serde_json::Map::new();
        for param in &self.parameters {
            values.insert(param.name.clone(), param.param_type.default_value());
        }
        values
    }
    
    /// 验证参数值是否满足约束
    pub fn validate_parameters(&self, values: &serde_json::Map<String, serde_json::Value>) -> Result<(), String> {
        // 验证类型和范围
        for param in &self.parameters {
            if let Some(value) = values.get(&param.name) {
                if !param.param_type.validate(value) {
                    return Err(format!("参数 '{}' 的值无效", param.name));
                }
            } else {
                return Err(format!("缺少参数 '{}'", param.name));
            }
        }
        
        // 验证约束
        for constraint in &self.constraints {
            match constraint {
                ParameterConstraint::Equal { params } => {
                    let first = params.first()
                        .and_then(|p| values.get(p))
                        .and_then(|v| v.as_f64());
                    if let Some(first_val) = first {
                        for param_name in params.iter().skip(1) {
                            let val = values.get(param_name).and_then(|v| v.as_f64());
                            if let Some(v) = val {
                                if (v - first_val).abs() > 1e-6 {
                                    return Err(format!("等式约束失败：{} 应该相等", params.join(", ")));
                                }
                            }
                        }
                    }
                }
                ParameterConstraint::Ratio { param1, param2, scale } => {
                    let v1 = values.get(param1).and_then(|v| v.as_f64());
                    let v2 = values.get(param2).and_then(|v| v.as_f64());
                    if let (Some(v1), Some(v2)) = (v1, v2) {
                        if (v1 - v2 * scale).abs() > 1e-6 {
                            return Err(format!("比例约束失败：{} = {} * {}", param1, param2, scale));
                        }
                    }
                }
                _ => {} // 其他约束类型暂不验证
            }
        }
        
        Ok(())
    }
}

/// 参数化块实例
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ParametricBlockInstance {
    /// 引用的块定义名称
    pub block_name: String,
    /// 插入点
    pub insertion_point: Point2,
    /// 旋转角度（度）
    pub rotation: f64,
    /// 参数值
    pub parameter_values: serde_json::Map<String, serde_json::Value>,
    /// 元数据
    pub metadata: EntityMetadata,
}

impl ParametricBlockInstance {
    /// 创建新的实例
    pub fn new(block_name: impl Into<String>, insertion_point: Point2) -> Self {
        Self {
            block_name: block_name.into(),
            insertion_point,
            rotation: 0.0,
            parameter_values: serde_json::Map::new(),
            metadata: EntityMetadata::default(),
        }
    }
    
    /// 设置参数值
    pub fn with_parameter(mut self, name: impl Into<String>, value: serde_json::Value) -> Self {
        self.parameter_values.insert(name.into(), value);
        self
    }
    
    /// 设置旋转角度
    pub fn with_rotation(mut self, rotation: f64) -> Self {
        self.rotation = rotation;
        self
    }
}

/// 路径命令 (SVG-like)
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "cmd", rename_all = "PascalCase")]
pub enum PathCommand {
    MoveTo { x: f64, y: f64 },
    LineTo { x: f64, y: f64 },
    ArcTo {
        rx: f64,
        ry: f64,
        x_axis_rotation: f64,
        large_arc: bool,
        sweep: bool,
        x: f64,
        y: f64,
    },
    Close,
}

/// 实体元数据
#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
pub struct EntityMetadata {
    pub layer: Option<String>,
    pub color: Option<String>,
    pub lineweight: Option<f64>,
    pub line_type: Option<String>,
    pub handle: Option<String>,
    /// 材料名称（从颜色/图层名映射）
    pub material: Option<String>,
    /// 宽度（用于门窗开口）
    pub width: Option<f64>,
}

impl EntityMetadata {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_layer(mut self, layer: impl Into<String>) -> Self {
        self.layer = Some(layer.into());
        self
    }

    pub fn with_material(mut self, material: impl Into<String>) -> Self {
        self.material = Some(material.into());
        self
    }

    pub fn with_width(mut self, width: f64) -> Self {
        self.width = Some(width);
        self
    }
}

/// RawEntity 辅助方法
impl RawEntity {
    /// 获取实体的语义标签
    pub fn semantic(&self) -> Option<&BoundarySemantic> {
        match self {
            RawEntity::Line { semantic, .. } => semantic.as_ref(),
            RawEntity::Polyline { semantic, .. } => semantic.as_ref(),
            RawEntity::Arc { semantic, .. } => semantic.as_ref(),
            RawEntity::Circle { semantic, .. } => semantic.as_ref(),
            RawEntity::Text { semantic, .. } => semantic.as_ref(),
            RawEntity::Path { semantic, .. } => semantic.as_ref(),
            RawEntity::BlockReference { semantic, .. } => semantic.as_ref(),
            RawEntity::Dimension { semantic, .. } => semantic.as_ref(),
            RawEntity::Hatch { semantic, .. } => semantic.as_ref(),
            RawEntity::XRef { semantic, .. } => semantic.as_ref(),
        }
    }

    /// 设置实体的语义标签
    pub fn set_semantic(&mut self, semantic: Option<BoundarySemantic>) {
        match self {
            RawEntity::Line { semantic: s, .. } => *s = semantic,
            RawEntity::Polyline { semantic: s, .. } => *s = semantic,
            RawEntity::Arc { semantic: s, .. } => *s = semantic,
            RawEntity::Circle { semantic: s, .. } => *s = semantic,
            RawEntity::Text { semantic: s, .. } => *s = semantic,
            RawEntity::Path { semantic: s, .. } => *s = semantic,
            RawEntity::BlockReference { semantic: s, .. } => *s = semantic,
            RawEntity::Dimension { semantic: s, .. } => *s = semantic,
            RawEntity::Hatch { semantic: s, .. } => *s = semantic,
            RawEntity::XRef { semantic: s, .. } => *s = semantic,
        }
    }

    /// 获取实体的图层名
    pub fn layer(&self) -> Option<&str> {
        match self {
            RawEntity::Line { metadata, .. } => metadata.layer.as_deref(),
            RawEntity::Polyline { metadata, .. } => metadata.layer.as_deref(),
            RawEntity::Arc { metadata, .. } => metadata.layer.as_deref(),
            RawEntity::Circle { metadata, .. } => metadata.layer.as_deref(),
            RawEntity::Text { metadata, .. } => metadata.layer.as_deref(),
            RawEntity::Path { metadata, .. } => metadata.layer.as_deref(),
            RawEntity::BlockReference { metadata, .. } => metadata.layer.as_deref(),
            RawEntity::Dimension { metadata, .. } => metadata.layer.as_deref(),
            RawEntity::Hatch { metadata, .. } => metadata.layer.as_deref(),
            RawEntity::XRef { metadata, .. } => metadata.layer.as_deref(),
        }
    }

    /// 获取实体的颜色
    pub fn color(&self) -> Option<&str> {
        match self {
            RawEntity::Line { metadata, .. } => metadata.color.as_deref(),
            RawEntity::Polyline { metadata, .. } => metadata.color.as_deref(),
            RawEntity::Arc { metadata, .. } => metadata.color.as_deref(),
            RawEntity::Circle { metadata, .. } => metadata.color.as_deref(),
            RawEntity::Text { metadata, .. } => metadata.color.as_deref(),
            RawEntity::Path { metadata, .. } => metadata.color.as_deref(),
            RawEntity::BlockReference { metadata, .. } => metadata.color.as_deref(),
            RawEntity::Dimension { metadata, .. } => metadata.color.as_deref(),
            RawEntity::Hatch { metadata, .. } => metadata.color.as_deref(),
            RawEntity::XRef { metadata, .. } => metadata.color.as_deref(),
        }
    }

    /// 获取实体的元数据
    pub fn metadata(&self) -> &EntityMetadata {
        match self {
            RawEntity::Line { metadata, .. } => metadata,
            RawEntity::Polyline { metadata, .. } => metadata,
            RawEntity::Arc { metadata, .. } => metadata,
            RawEntity::Circle { metadata, .. } => metadata,
            RawEntity::Text { metadata, .. } => metadata,
            RawEntity::Path { metadata, .. } => metadata,
            RawEntity::BlockReference { metadata, .. } => metadata,
            RawEntity::Dimension { metadata, .. } => metadata,
            RawEntity::Hatch { metadata, .. } => metadata,
            RawEntity::XRef { metadata, .. } => metadata,
        }
    }

    /// 获取实体类型名称
    pub fn entity_type_name(&self) -> &'static str {
        match self {
            RawEntity::Line { .. } => "Line",
            RawEntity::Polyline { .. } => "Polyline",
            RawEntity::Arc { .. } => "Arc",
            RawEntity::Circle { .. } => "Circle",
            RawEntity::Text { .. } => "Text",
            RawEntity::Path { .. } => "Path",
            RawEntity::BlockReference { .. } => "BlockReference",
            RawEntity::Dimension { .. } => "Dimension",
            RawEntity::Hatch { .. } => "Hatch",
            RawEntity::XRef { .. } => "XRef",
        }
    }

    /// 获取实体的句柄（如果有）
    pub fn handle(&self) -> Option<&str> {
        self.metadata().handle.as_deref()
    }
}

/// 计算 2D 点的欧几里得距离
#[inline]
pub fn distance_2d(a: Point2, b: Point2) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    (dx * dx + dy * dy).sqrt()
}

/// 计算向量叉积 (用于方向判断)
#[inline]
pub fn cross_product_2d(o: Point2, a: Point2, b: Point2) -> f64 {
    (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
}

/// 快速平方距离计算 (避免开方)
#[inline]
pub fn distance_squared_2d(a: Point2, b: Point2) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    dx * dx + dy * dy
}

// ============================================================================
// PDF 光栅图像类型
// ============================================================================

/// PDF 中的光栅图像
#[derive(Debug, Clone)]
pub struct PdfRasterImage {
    /// 图像名称
    pub name: String,
    /// 宽度（像素）
    pub width: u32,
    /// 高度（像素）
    pub height: u32,
    /// 原始像素数据（RGBA）
    pub pixels: Vec<u8>,
    /// DPI 信息（如果有）
    pub dpi: Option<(f64, f64)>,
    /// 在 PDF 中的变换矩阵
    pub transform: [f64; 6],
}

impl PdfRasterImage {
    /// 创建新的 PdfRasterImage
    pub fn new(
        name: String,
        width: u32,
        height: u32,
        pixels: Vec<u8>,
        dpi: Option<(f64, f64)>,
        transform: [f64; 6],
    ) -> Self {
        Self {
            name,
            width,
            height,
            pixels,
            dpi,
            transform,
        }
    }

    /// 转换为 DynamicImage
    pub fn to_image(&self) -> image::DynamicImage {
        if self.pixels.len() == (self.width * self.height) as usize {
            // 灰度图像
            image::DynamicImage::ImageLuma8(
                image::GrayImage::from_raw(self.width, self.height, self.pixels.clone())
                    .unwrap_or_else(|| image::GrayImage::new(self.width, self.height))
            )
        } else if self.pixels.len() == (self.width * self.height * 3) as usize {
            // RGB 图像
            image::DynamicImage::ImageRgb8(
                image::RgbImage::from_raw(self.width, self.height, self.pixels.clone())
                    .unwrap_or_else(|| image::RgbImage::new(self.width, self.height))
            )
        } else if self.pixels.len() == (self.width * self.height * 4) as usize {
            // RGBA 图像
            image::DynamicImage::ImageRgba8(
                image::RgbaImage::from_raw(self.width, self.height, self.pixels.clone())
                    .unwrap_or_else(|| image::RgbaImage::new(self.width, self.height))
            )
        } else {
            // 数据不匹配，创建空白图像
            image::DynamicImage::new_rgb8(self.width, self.height)
        }
    }

    /// 获取 DPI（如果有）
    pub fn dpi(&self) -> Option<(f64, f64)> {
        self.dpi
    }

    /// 获取变换矩阵
    pub fn transform(&self) -> &[f64; 6] {
        &self.transform
    }
}

/// 解码图像像素数据（供 parser crate 使用）
pub fn decode_image_pixels(data: &[u8], _width: u32, _height: u32) -> Vec<u8> {
    // 尝试使用 image crate 解码
    if let Ok(img) = image::load_from_memory(data) {
        return img.to_rgba8().into_raw();
    }
    
    // 如果解码失败，返回原始数据（可能是裸像素）
    data.to_vec()
}

/// 从 parser crate 的 RasterImage 参数创建 PdfRasterImage
/// 
/// 这是一个辅助函数，用于避免 common-types 和 parser 之间的循环依赖
pub fn pdf_raster_from_parser_raster(
    name: String,
    width: u32,
    height: u32,
    data: &[u8],
    dpi_x: f64,
    dpi_y: f64,
) -> PdfRasterImage {
    let pixels = decode_image_pixels(data, width, height);
    
    PdfRasterImage {
        name,
        width,
        height,
        pixels,
        dpi: Some((dpi_x, dpi_y)),
        transform: [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_distance_2d() {
        let a = [0.0, 0.0];
        let b = [3.0, 4.0];
        assert!((distance_2d(a, b) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_cross_product_2d() {
        let o = [0.0, 0.0];
        let a = [1.0, 0.0];
        let b = [0.0, 1.0];
        assert!((cross_product_2d(o, a, b) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_distance_squared_2d() {
        let a = [0.0, 0.0];
        let b = [3.0, 4.0];
        assert!((distance_squared_2d(a, b) - 25.0).abs() < 1e-10);
    }
}

// ============================================================================
// 参数化块系统测试
// ============================================================================

#[cfg(test)]
mod parametric_block_tests {
    use super::*;

    fn create_door_block() -> ParametricBlockDefinition {
        ParametricBlockDefinition::new("DOOR", [0.0, 0.0])
            .add_parameter(ParameterDefinition {
                name: "width".into(),
                param_type: ParameterType::Length {
                    default: 900.0,
                    min: Some(600.0),
                    max: Some(1200.0),
                },
                description: Some("门宽度".into()),
                group: Some("尺寸".into()),
                visible: true,
            })
            .add_parameter(ParameterDefinition {
                name: "height".into(),
                param_type: ParameterType::Length {
                    default: 1800.0,  // 修改为宽度的 2 倍
                    min: Some(1200.0),
                    max: Some(2400.0),
                },
                description: Some("门高度".into()),
                group: Some("尺寸".into()),
                visible: true,
            })
            .add_constraint(ParameterConstraint::Ratio {
                param1: "height".into(),
                param2: "width".into(),
                scale: 2.0,
            })
    }

    #[test]
    fn test_parameter_type_validation() {
        let length_param = ParameterType::Length {
            default: 100.0,
            min: Some(0.0),
            max: Some(200.0),
        };

        assert!(length_param.validate(&serde_json::json!(100.0)));
        assert!(!length_param.validate(&serde_json::json!(300.0))); // 超出最大值
        assert!(!length_param.validate(&serde_json::json!(-10.0))); // 低于最小值

        let bool_param = ParameterType::Boolean { default: true };
        assert!(bool_param.validate(&serde_json::json!(true)));
        assert!(!bool_param.validate(&serde_json::json!("true"))); // 类型错误

        let enum_param = ParameterType::Enum {
            default: "A".into(),
            options: vec!["A".into(), "B".into(), "C".into()],
        };
        assert!(enum_param.validate(&serde_json::json!("A")));
        assert!(!enum_param.validate(&serde_json::json!("D"))); // 不在选项中
    }

    #[test]
    fn test_parametric_block_creation() {
        let block = create_door_block();

        assert_eq!(block.name, "DOOR");
        assert_eq!(block.parameters.len(), 2);
        assert_eq!(block.constraints.len(), 1);

        let defaults = block.default_parameter_values();
        assert_eq!(defaults.get("width").unwrap().as_f64(), Some(900.0));
        assert_eq!(defaults.get("height").unwrap().as_f64(), Some(1800.0));  // 修改为 1800.0
    }

    #[test]
    fn test_parameter_validation() {
        let block = create_door_block();

        // 有效参数
        let valid_values = serde_json::json!({
            "width": 800.0,
            "height": 1600.0,
        })
        .as_object()
        .unwrap()
        .clone();

        assert!(block.validate_parameters(&valid_values).is_ok());

        // 无效参数（超出范围）
        let invalid_values = serde_json::json!({
            "width": 500.0, // 低于最小值
            "height": 2100.0,
        })
        .as_object()
        .unwrap()
        .clone();

        assert!(block.validate_parameters(&invalid_values).is_err());
    }

    #[test]
    fn test_constraint_validation() {
        let block = create_door_block();

        // 满足约束
        let valid_values = serde_json::json!({
            "width": 800.0,
            "height": 1600.0, // 800 * 2 = 1600
        })
        .as_object()
        .unwrap()
        .clone();

        assert!(block.validate_parameters(&valid_values).is_ok());

        // 不满足约束
        let invalid_values = serde_json::json!({
            "width": 800.0,
            "height": 1500.0, // 800 * 2 != 1500
        })
        .as_object()
        .unwrap()
        .clone();

        // 注意：当前约束验证只检查等式和比例约束
        // 这里应该失败，但由于实现限制可能不会
        let result = block.validate_parameters(&invalid_values);
        // 比例约束检查应该失败
        assert!(result.is_err());
    }

    #[test]
    fn test_parametric_block_instance() {
        let instance = ParametricBlockInstance::new("DOOR", [0.0, 0.0])
            .with_parameter("width", serde_json::json!(900.0))
            .with_parameter("height", serde_json::json!(1800.0))
            .with_rotation(90.0);

        assert_eq!(instance.block_name, "DOOR");
        assert_eq!(instance.insertion_point, [0.0, 0.0]);
        assert_eq!(instance.rotation, 90.0);
        assert_eq!(instance.parameter_values.get("width").unwrap().as_f64(), Some(900.0));
    }
}
