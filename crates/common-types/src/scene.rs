//! 场景状态定义
//!
//! 用于在交互服务和验证服务间传递的完整场景描述

use crate::geometry::Point2;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

// ============================================================================
// 解析进度跟踪（P0-1: 加载进度可视化）
// ============================================================================

/// 解析阶段枚举
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[derive(Default)]
pub enum ParseStage {
    /// 读取文件
    #[default]
    ReadingFile,
    /// 解析文件头
    ParsingHeader,
    /// 解析表格段
    ParsingTables,
    /// 解析块定义
    ParsingBlocks,
    /// 解析实体
    ParsingEntities,
    /// 构建拓扑
    BuildingTopology,
    /// 几何计算
    ComputingGeometry,
    /// 收尾处理
    Finalizing,
}

impl ParseStage {
    /// 获取中文显示名称
    pub fn name_zh(&self) -> &'static str {
        match self {
            ParseStage::ReadingFile => "读取文件",
            ParseStage::ParsingHeader => "解析文件头",
            ParseStage::ParsingTables => "解析表格段",
            ParseStage::ParsingBlocks => "解析块定义",
            ParseStage::ParsingEntities => "解析实体",
            ParseStage::BuildingTopology => "构建拓扑",
            ParseStage::ComputingGeometry => "几何计算",
            ParseStage::Finalizing => "收尾处理",
        }
    }

    /// 获取当前阶段的进度权重（用于计算总体进度）
    pub fn weight(&self) -> f64 {
        match self {
            ParseStage::ReadingFile => 0.05,
            ParseStage::ParsingHeader => 0.05,
            ParseStage::ParsingTables => 0.10,
            ParseStage::ParsingBlocks => 0.15,
            ParseStage::ParsingEntities => 0.35,
            ParseStage::BuildingTopology => 0.15,
            ParseStage::ComputingGeometry => 0.10,
            ParseStage::Finalizing => 0.05,
        }
    }
}

/// 解析进度信息
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ParseProgress {
    /// 当前阶段
    pub stage: ParseStage,
    /// 已解析实体数
    pub entities_parsed: usize,
    /// 预估总实体数（可选，解析完成后才知道）
    pub total_entities: Option<usize>,
    /// 已读取字节数
    pub bytes_read: u64,
    /// 文件总字节数
    pub total_bytes: u64,
    /// 阶段内进度（0.0 - 1.0）
    pub stage_progress: f64,
    /// 预估剩余时间（秒）
    pub eta_seconds: Option<f64>,
    /// 总体进度（0.0 - 1.0，计算得出）
    pub overall_progress: f64,
}

impl ParseProgress {
    /// 创建新的进度信息
    pub fn new(stage: ParseStage, total_bytes: u64) -> Self {
        Self {
            stage,
            total_bytes,
            stage_progress: 0.0,
            overall_progress: 0.0,
            ..Default::default()
        }
    }

    /// 计算总体进度
    pub fn compute_overall_progress(&mut self) -> f64 {
        let stage_weight = self.stage.weight();

        // 计算之前所有阶段的累计权重
        let previous_weight = match self.stage {
            ParseStage::ReadingFile => 0.0,
            ParseStage::ParsingHeader => 0.05,
            ParseStage::ParsingTables => 0.10,
            ParseStage::ParsingBlocks => 0.20,
            ParseStage::ParsingEntities => 0.35,
            ParseStage::BuildingTopology => 0.50,
            ParseStage::ComputingGeometry => 0.65,
            ParseStage::Finalizing => 0.75,
        };

        // 当前阶段内的进度贡献
        let stage_contribution = stage_weight * self.stage_progress;

        self.overall_progress = (previous_weight + stage_contribution).min(1.0);
        self.overall_progress
    }

    /// 获取进度百分比字符串
    pub fn progress_percent(&self) -> String {
        format!("{:.1}%", self.overall_progress * 100.0)
    }

    /// 获取状态描述字符串
    pub fn status_message(&self) -> String {
        let stage_name = self.stage.name_zh();
        if let Some(total) = self.total_entities {
            if total > 0 {
                return format!(
                    "{} - {}/{} 实体 ({})",
                    stage_name,
                    self.entities_parsed,
                    total,
                    self.progress_percent()
                );
            }
        }
        format!("{} ({})", stage_name, self.progress_percent())
    }
}

// ============================================================================
// 基础类型定义（声学分析依赖）
// ============================================================================

/// 表面 ID（边 ID 的别名）
pub type SurfaceId = usize;

/// 材料定义（用于声学分析）
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct Material {
    /// 材料名称
    pub name: String,
    /// 吸声系数（频率相关）
    ///
    /// 典型值参考（500Hz）：
    /// - 混凝土：0.02-0.03
    /// - 砖墙：0.03-0.05
    /// - 石膏板：0.05-0.10
    /// - 木材：0.10-0.15
    /// - 地毯：0.30-0.50
    /// - 玻璃：0.03-0.10
    /// - 布艺座椅：0.60-0.80
    #[serde(default)]
    pub absorption_coeffs: BTreeMap<crate::acoustic::Frequency, f64>,
    /// 散射系数（可选）
    #[serde(default)]
    pub scattering_coefficient: f64,
}

impl Default for Material {
    fn default() -> Self {
        Self {
            name: "default".to_string(),
            absorption_coeffs: BTreeMap::new(),
            scattering_coefficient: 0.0,
        }
    }
}

impl Material {
    /// 创建新材料
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            ..Default::default()
        }
    }

    /// 设置吸声系数
    pub fn with_absorption(mut self, freq: crate::acoustic::Frequency, coeff: f64) -> Self {
        self.absorption_coeffs.insert(freq, coeff);
        self
    }

    /// 设置默认吸声系数（所有频率）
    pub fn with_default_absorption(mut self, coeff: f64) -> Self {
        for freq in crate::acoustic::Frequency::all() {
            self.absorption_coeffs.insert(freq, coeff);
        }
        self
    }

    /// 获取 500Hz 的吸声系数（常用参考频率）
    pub fn absorption_at_500hz(&self) -> f64 {
        self.absorption_coeffs
            .get(&crate::acoustic::Frequency::Hz500)
            .copied()
            .unwrap_or(0.0)
    }
}

// ============================================================================
// 场景状态定义
// ============================================================================

/// 场景状态 - 核心数据结构
#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
pub struct SceneState {
    /// 外轮廓 (正方向)
    pub outer: Option<ClosedLoop>,
    /// 孔洞列表 (负方向)
    pub holes: Vec<ClosedLoop>,
    /// 边界语义标注
    pub boundaries: Vec<BoundarySegment>,
    /// 声源列表 (可选)
    pub sources: Vec<SoundSource>,
    /// 原始边数据（用于前端显示）
    #[serde(default)]
    pub edges: Vec<RawEdge>,
    /// 单位
    pub units: LengthUnit,
    /// 坐标系描述
    pub coordinate_system: CoordinateSystem,

    // ========================================================================
    // 座椅区域与 LOD 渲染支持（P11 技术设计文档 v1.0）
    // ========================================================================
    /// 座椅区域列表
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub seat_zones: Vec<SeatZone>,

    /// 渲染配置
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub render_config: Option<RenderConfig>,
}

/// 原始边（用于前端显示）
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct RawEdge {
    /// 边 ID
    pub id: usize,
    /// 起点 [x, y]
    pub start: [f64; 2],
    /// 终点 [x, y]
    pub end: [f64; 2],
    /// 图层名称（可选）
    #[serde(default)]
    pub layer: Option<String>,
    /// 颜色索引（可选）
    #[serde(default)]
    pub color_index: Option<u16>,
}

/// 闭合环
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ClosedLoop {
    /// 点序列 (首尾相连)
    pub points: Vec<Point2>,
    /// 有符号面积 (>0 为外轮廓，<0 为孔洞)
    pub signed_area: f64,
}

impl ClosedLoop {
    pub fn new(points: Vec<Point2>) -> Self {
        let signed_area = calculate_signed_area(&points);
        Self {
            points,
            signed_area,
        }
    }

    pub fn is_outer(&self) -> bool {
        self.signed_area > 0.0
    }

    pub fn is_hole(&self) -> bool {
        self.signed_area < 0.0
    }
}

/// 边界段语义
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct BoundarySegment {
    /// 线段起点索引 (在 outer 或 holes 中)
    pub segment: [usize; 2],
    /// 语义类型
    pub semantic: BoundarySemantic,
    /// 材料名称 (可选)
    pub material: Option<String>,
    /// 宽度 (仅用于开口)
    pub width: Option<f64>,
}

impl BoundarySegment {
    /// 从图层名和材料自动推断边界段语义
    ///
    /// # Arguments
    /// * `layer_name` - DXF 图层名
    /// * `material` - 材料名称（可选）
    /// * `width` - 宽度（仅用于开口，可选）
    ///
    /// # Returns
    /// 自动推断的 BoundarySegment
    pub fn infer_from_layer(
        layer_name: &str,
        material: Option<String>,
        width: Option<f64>,
    ) -> Self {
        let semantic = Self::infer_semantic_from_layer(layer_name);

        Self {
            segment: [0, 0], // 需要后续设置
            semantic,
            material,
            width,
        }
    }

    /// 根据图层名推断语义类型
    ///
    /// # 推断规则
    /// 1. 门图层：包含 "DOOR", "门" 等关键词 → `BoundarySemantic::Door`
    /// 2. 窗图层：包含 "WINDOW", "窗" 等关键词 → `BoundarySemantic::Window`
    /// 3. 开口图层：包含 "OPEN", "开口" 等关键词 → `BoundarySemantic::Opening`
    /// 4. 墙体图层：包含 "WALL", "墙" 等关键词 → `BoundarySemantic::HardWall`
    /// 5. 默认 → `BoundarySemantic::HardWall`
    pub fn infer_semantic_from_layer(layer_name: &str) -> BoundarySemantic {
        let upper = layer_name.to_uppercase();

        // 门图层（优先于窗）
        let door_keywords = [
            "DOOR",
            "门",
            "DOORS",
            "入户门",
            "室内门",
            "防火门",
            "单开门",
            "双开门",
            "推拉门",
        ];
        if door_keywords.iter().any(|k| upper.contains(k)) {
            return BoundarySemantic::Door;
        }

        // 门模式匹配
        if upper.starts_with("A-DOOR")
            || upper.contains("DOOR-")
            || upper.contains("DOOR_")
            || upper.starts_with("D-")
        {
            return BoundarySemantic::Door;
        }

        // 窗图层
        let window_keywords = [
            "WINDOW",
            "窗",
            "WINDOWS",
            "GLAZ",
            "GLASS",
            "采光窗",
            "天窗",
            "落地窗",
            "百叶窗",
        ];
        if window_keywords.iter().any(|k| upper.contains(k)) {
            return BoundarySemantic::Window;
        }

        // 窗模式匹配
        if upper.starts_with("A-WIND")
            || upper.contains("WINDOW-")
            || upper.contains("WINDOW_")
            || upper.contains("GLAZ-")
            || upper.contains("GLASS-")
            || upper.starts_with("W-")
        {
            return BoundarySemantic::Window;
        }

        // 开口图层
        let opening_keywords = ["OPEN", "开口", "OPENING", "HOLE", "洞", "GATE", "通道"];
        if opening_keywords.iter().any(|k| upper.contains(k)) {
            return BoundarySemantic::Opening;
        }

        // 开口模式匹配
        if upper.starts_with("A-OPEN") || upper.contains("OPENING-") || upper.contains("OPENING_") {
            return BoundarySemantic::Opening;
        }

        // 墙体图层
        let wall_keywords = [
            "WALL",
            "墙",
            "WALLS",
            "墙体",
            "内墙",
            "外墙",
            "剪力墙",
            "隔墙",
            "STRUCT",
            "结构",
            "COLUMN",
            "柱",
            "BEAM",
            "梁",
            "STRUC",
        ];
        if wall_keywords.iter().any(|k| upper.contains(k)) {
            return BoundarySemantic::HardWall;
        }

        // AIA 标准墙体模式
        if upper.starts_with("A-WALL")
            || upper.starts_with("S-WALL")
            || upper.starts_with("S-STRC")
            || upper.starts_with("A-COLS")
        {
            return BoundarySemantic::HardWall;
        }

        // 默认硬墙
        BoundarySemantic::HardWall
    }

    /// 根据颜色索引推断材料
    ///
    /// # ACI 颜色与材料映射
    /// - 1 (红色): 混凝土 (concrete)
    /// - 2 (黄色): 砖墙 (brick)
    /// - 3 (绿色): 木材 (wood)
    /// - 4 (青色): 石膏板 (gypsum)
    /// - 5 (蓝色): 玻璃 (glass)
    /// - 6 (洋红): 金属 (metal)
    /// - 7 (黑色/白色): 默认墙体 (default_wall)
    pub fn infer_material_from_aci_color(color_index: u16) -> Option<String> {
        match color_index {
            1 => Some("concrete".to_string()),     // 红色=混凝土
            2 => Some("brick".to_string()),        // 黄色=砖墙
            3 => Some("wood".to_string()),         // 绿色=木材
            4 => Some("gypsum".to_string()),       // 青色=石膏板
            5 => Some("glass".to_string()),        // 蓝色=玻璃
            6 => Some("metal".to_string()),        // 洋红=金属
            7 => Some("default_wall".to_string()), // 黑色/白色=默认墙体
            _ => None,
        }
    }

    /// 根据图层名推断材料（备选方案，当颜色信息缺失时使用）
    pub fn infer_material_from_layer(layer_name: &str) -> Option<String> {
        let upper = layer_name.to_uppercase();

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
        if upper.contains("METAL")
            || upper.contains("STEEL")
            || upper.contains("钢")
            || upper.contains("金属")
        {
            return Some("metal".to_string());
        }

        // 石膏板
        if upper.contains("GYPSUM") || upper.contains("石膏") {
            return Some("gypsum".to_string());
        }

        None
    }

    /// 计算开口宽度（仅用于门/窗/开口）
    ///
    /// # Arguments
    /// * `start` - 起点坐标
    /// * `end` - 终点坐标
    ///
    /// # Returns
    /// 宽度（米），如果不是开口类型则返回 None
    pub fn calculate_width(start: Point2, end: Point2) -> Option<f64> {
        let dx = end[0] - start[0];
        let dy = end[1] - start[1];
        let width_mm = (dx * dx + dy * dy).sqrt();

        // 转换为米
        let width_m = width_mm / 1000.0;

        // 合理的开口宽度范围：0.5m - 5.0m
        if (0.5..=5.0).contains(&width_m) {
            Some(width_m)
        } else {
            None
        }
    }
}

/// 边界语义类型
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum BoundarySemantic {
    /// 硬墙
    HardWall,
    /// 吸声墙
    AbsorptiveWall,
    /// 开口/门洞
    Opening,
    /// 窗户
    Window,
    /// 门
    Door,
    /// 家具
    Furniture,
    /// 卫浴设备
    BathroomFixture,
    /// 厨房设备
    KitchenFixture,
    /// 暖通空调
    Hvac,
    /// 电气/照明
    Electrical,
    /// 标注/文字
    Annotation,
    /// 自定义
    Custom(String),
}

/// 声源类型
#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum SourceType {
    /// 全向声源
    #[default]
    Omnidirectional,
    /// 心形指向
    Cardioid,
    /// 超心形指向
    HyperCardioid,
    /// 8 字形指向
    Figure8,
    /// 自定义指向性
    Custom,
}

/// 声源
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct SoundSource {
    /// 声源 ID
    pub id: String,
    /// 位置 (x, y, z)
    pub position: [f64; 3],
    /// 朝向 [yaw, pitch]（可选）
    pub orientation: Option<[f64; 2]>,
    /// 指向性类型
    pub source_type: SourceType,
    /// 增益 (dB)
    pub gain_db: f64,
    /// 延迟 (ms)
    pub delay_ms: f64,
}

impl Default for SoundSource {
    fn default() -> Self {
        Self {
            id: String::new(),
            position: [0.0, 0.0, 0.0],
            orientation: None,
            source_type: SourceType::default(),
            gain_db: 0.0,
            delay_ms: 0.0,
        }
    }
}

/// 长度单位
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum LengthUnit {
    Mm,
    Cm,
    M,
    Inch,
    Foot,
    /// 码（英制，1 yard = 3 feet = 0.9144 米）
    Yard,
    /// 英里（英制，1 mile = 5280 feet = 1609.344 米）
    Mile,
    /// 微米（1 微米 = 0.001 毫米）
    Micron,
    /// 千米/公里（1 千米 = 1000 米）
    Kilometer,
    /// 点（印刷单位，1 point = 1/72 英寸 ≈ 0.3528 毫米）
    Point,
    /// 派卡（印刷单位，1 pica = 12 points ≈ 4.233 毫米）
    Pica,
    #[default]
    Unspecified,
}

impl LengthUnit {
    /// 转换到米
    pub fn to_meters(&self, value: f64) -> f64 {
        match self {
            LengthUnit::Mm => value / 1000.0,
            LengthUnit::Cm => value / 100.0,
            LengthUnit::M => value,
            LengthUnit::Inch => value * 0.0254,
            LengthUnit::Foot => value * 0.3048,
            LengthUnit::Yard => value * 0.9144,
            LengthUnit::Mile => value * 1609.344,
            LengthUnit::Micron => value / 1_000_000.0,
            LengthUnit::Kilometer => value * 1000.0,
            LengthUnit::Point => value * 0.0254 / 72.0,
            LengthUnit::Pica => value * 0.0254 / 6.0,
            LengthUnit::Unspecified => value, // 需要用户标定
        }
    }

    /// 从米转换
    pub fn from_meters(&self, value: f64) -> f64 {
        match self {
            LengthUnit::Mm => value * 1000.0,
            LengthUnit::Cm => value * 100.0,
            LengthUnit::M => value,
            LengthUnit::Inch => value / 0.0254,
            LengthUnit::Foot => value / 0.3048,
            LengthUnit::Yard => value / 0.9144,
            LengthUnit::Mile => value / 1609.344,
            LengthUnit::Micron => value * 1_000_000.0,
            LengthUnit::Kilometer => value / 1000.0,
            LengthUnit::Point => value * 72.0 / 0.0254,
            LengthUnit::Pica => value * 6.0 / 0.0254,
            LengthUnit::Unspecified => value,
        }
    }

    /// 转换到毫米
    pub fn to_mm(&self, value: f64) -> f64 {
        match self {
            LengthUnit::Mm => value,
            LengthUnit::Cm => value * 10.0,
            LengthUnit::M => value * 1000.0,
            LengthUnit::Inch => value * 25.4,
            LengthUnit::Foot => value * 304.8,
            LengthUnit::Yard => value * 914.4,
            LengthUnit::Mile => value * 1_609_344.0,
            LengthUnit::Micron => value / 1000.0,
            LengthUnit::Kilometer => value * 1_000_000.0,
            LengthUnit::Point => value * 0.352778,
            LengthUnit::Pica => value * 4.23333,
            LengthUnit::Unspecified => value,
        }
    }

    /// 从毫米转换
    pub fn from_mm(&self, value: f64) -> f64 {
        match self {
            LengthUnit::Mm => value,
            LengthUnit::Cm => value / 10.0,
            LengthUnit::M => value / 1000.0,
            LengthUnit::Inch => value / 25.4,
            LengthUnit::Foot => value / 304.8,
            LengthUnit::Yard => value / 914.4,
            LengthUnit::Mile => value / 1_609_344.0,
            LengthUnit::Micron => value * 1000.0,
            LengthUnit::Kilometer => value / 1_000_000.0,
            LengthUnit::Point => value / 0.352778,
            LengthUnit::Pica => value / 4.23333,
            LengthUnit::Unspecified => value,
        }
    }

    /// 获取单位名称（中文）
    pub fn name_zh(&self) -> &'static str {
        match self {
            LengthUnit::Mm => "毫米",
            LengthUnit::Cm => "厘米",
            LengthUnit::M => "米",
            LengthUnit::Inch => "英寸",
            LengthUnit::Foot => "英尺",
            LengthUnit::Yard => "码",
            LengthUnit::Mile => "英里",
            LengthUnit::Micron => "微米",
            LengthUnit::Kilometer => "千米",
            LengthUnit::Point => "点",
            LengthUnit::Pica => "派卡",
            LengthUnit::Unspecified => "未指定",
        }
    }

    /// 获取单位名称（英文）
    pub fn name_en(&self) -> &'static str {
        match self {
            LengthUnit::Mm => "Millimeter",
            LengthUnit::Cm => "Centimeter",
            LengthUnit::M => "Meter",
            LengthUnit::Inch => "Inch",
            LengthUnit::Foot => "Foot",
            LengthUnit::Yard => "Yard",
            LengthUnit::Mile => "Mile",
            LengthUnit::Micron => "Micron",
            LengthUnit::Kilometer => "Kilometer",
            LengthUnit::Point => "Point",
            LengthUnit::Pica => "Pica",
            LengthUnit::Unspecified => "Unspecified",
        }
    }

    /// 从 DXF $INSUNITS 代码创建单位
    ///
    /// DXF $INSUNITS 代码定义：
    /// - 0: Unspecified
    /// - 1: Inches
    /// - 2: Feet
    /// - 3: Miles
    /// - 4: Millimeters
    /// - 5: Centimeters
    /// - 6: Meters
    /// - 7: Microinches (微英寸，10^-6 英寸)
    /// - 8: Mils (密耳，10^-3 英寸)
    /// - 9: Yards
    /// - 10: Angstroms (埃，10^-10 米)
    /// - 11: Nanometers (纳米)
    /// - 12: Microns (微米)
    /// - 13: Decimeters (分米)
    /// - 14: Dekameters (十米)
    /// - 15: Hectometers (百米)
    /// - 16: Gigameters (吉米)
    /// - 17: Astronomical units (天文单位)
    /// - 18: Light years (光年)
    /// - 19: Parsecs (秒差距)
    pub fn from_dxf_code(code: i32) -> Self {
        match code {
            0 => LengthUnit::Unspecified,
            1 => LengthUnit::Inch,
            2 => LengthUnit::Foot,
            3 => LengthUnit::Mile,
            4 => LengthUnit::Mm,
            5 => LengthUnit::Cm,
            6 => LengthUnit::M,
            7 => LengthUnit::Micron, // 微英寸，近似为微米
            8 => LengthUnit::Micron, // 密耳，近似为微米
            9 => LengthUnit::Yard,
            10 => LengthUnit::Micron, // 埃，近似为微米
            11 => LengthUnit::Micron, // 纳米，近似为微米
            12 => LengthUnit::Micron,
            13 => LengthUnit::Cm,             // 分米，近似为厘米
            14 => LengthUnit::M,              // 十米，近似为米
            15 => LengthUnit::M,              // 百米，近似为米
            16 => LengthUnit::Kilometer,      // 吉米，近似为千米
            17..=19 => LengthUnit::Kilometer, // 天文单位/光年/秒差距，近似为千米
            _ => LengthUnit::Unspecified,
        }
    }

    /// 转换为 DXF $INSUNITS 代码
    pub fn to_dxf_code(&self) -> i32 {
        match self {
            LengthUnit::Unspecified => 0,
            LengthUnit::Inch => 1,
            LengthUnit::Foot => 2,
            LengthUnit::Mile => 3,
            LengthUnit::Mm => 4,
            LengthUnit::Cm => 5,
            LengthUnit::M => 6,
            LengthUnit::Micron => 12,
            LengthUnit::Yard => 9,
            LengthUnit::Kilometer => 16,
            LengthUnit::Point => 4, // 点，近似为毫米
            LengthUnit::Pica => 4,  // 派卡，近似为毫米
        }
    }
}

/// 坐标系类型
#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum CoordinateSystem {
    #[default]
    RightHandedYUp,
    RightHandedZUp,
    LeftHandedYUp,
    LeftHandedZUp,
}

// ============================================================================
// 座椅区域与 LOD 渲染支持（P11 技术设计文档 v1.0）
// ============================================================================

/// 座椅类型（基于间距和布局推断）
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum SeatType {
    /// 单人椅
    Single,
    /// 双人椅
    Double,
    /// 礼堂排椅
    Auditorium,
    /// 长凳
    Bench,
    /// 未知类型
    #[default]
    Unknown,
}

/// 声学属性（用于声学仿真）
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct AcousticProps {
    /// 吸声系数 (0.0-1.0)
    ///
    /// 典型值参考：
    /// - 空座椅（硬面）：0.20-0.30
    /// - 空座椅（软垫）：0.50-0.70
    /// - 礼堂椅（布艺）：0.60-0.80
    /// - 观众（坐满）：0.80-0.95
    #[serde(default = "default_absorption")]
    pub absorption_coefficient: f32,
    /// 散射系数 (0.0-1.0)
    #[serde(default = "default_scattering")]
    pub scattering_coefficient: f32,
}

fn default_absorption() -> f32 {
    0.3
}
fn default_scattering() -> f32 {
    0.1
}

impl Default for AcousticProps {
    fn default() -> Self {
        Self {
            absorption_coefficient: default_absorption(),
            scattering_coefficient: default_scattering(),
        }
    }
}

/// 座椅区域（声学仿真简化表示）
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct SeatZone {
    /// 座椅区域边界（凸包简化）
    pub boundary: ClosedLoop,
    /// 座椅数量
    pub seat_count: usize,
    /// 座椅类型
    pub seat_type: SeatType,
    /// 声学属性
    pub acoustic_properties: AcousticProps,
    /// 原始座椅插入点（可选，用于 LOD2 渲染）
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub original_positions: Option<Vec<Point2>>,
}

/// 渲染细节级别（LOD）
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum LodLevel {
    /// 简化（色块表示）
    Simplified,
    /// 中等（阵列采样）
    Medium,
    /// 完整（逐个渲染）
    #[default]
    Detailed,
}

impl LodLevel {
    /// 根据座椅数量和缩放级别自动选择 LOD 级别
    pub fn auto_select(seat_count: usize, zoom_level: f64) -> Self {
        // 大规模场景或远距离查看时使用简化
        if seat_count > 500 || zoom_level < 0.1 {
            LodLevel::Simplified
        } else if seat_count > 50 || zoom_level < 0.5 {
            LodLevel::Medium
        } else {
            LodLevel::Detailed
        }
    }
}

/// 渲染配置
#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
pub struct RenderConfig {
    /// 推荐 LOD 级别
    pub recommended_lod: LodLevel,
    /// 座椅渲染阈值
    pub seat_render_threshold: usize,
    /// 是否启用 LOD 自动切换
    pub auto_lod: bool,
}

impl RenderConfig {
    /// 创建默认的 RenderConfig
    pub fn new(seat_count: usize) -> Self {
        Self {
            recommended_lod: LodLevel::auto_select(seat_count, 1.0),
            seat_render_threshold: 500,
            auto_lod: true,
        }
    }
}

/// 计算多边形的有符号面积
fn calculate_signed_area(points: &[Point2]) -> f64 {
    if points.len() < 3 {
        return 0.0;
    }

    let mut area = 0.0;
    let n = points.len();

    for i in 0..n {
        let j = (i + 1) % n;
        area += points[i][0] * points[j][1];
        area -= points[j][0] * points[i][1];
    }

    area / 2.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_signed_area_rectangle() {
        // 逆时针矩形 (正面积)
        let rect_cw = vec![[0.0, 0.0], [10.0, 0.0], [10.0, 8.0], [0.0, 8.0]];
        let loop_cw = ClosedLoop::new(rect_cw);
        assert!(loop_cw.is_outer());
        assert!((loop_cw.signed_area - 80.0).abs() < 1e-10);

        // 顺时针矩形 (负面积)
        let rect_ccw = vec![[0.0, 0.0], [0.0, 8.0], [10.0, 8.0], [10.0, 0.0]];
        let loop_ccw = ClosedLoop::new(rect_ccw);
        assert!(loop_ccw.is_hole());
    }

    // ========================================================================
    // 语义推断测试
    // ========================================================================

    #[test]
    fn test_infer_semantic_door() {
        // 门关键词
        assert_eq!(
            BoundarySegment::infer_semantic_from_layer("DOOR"),
            BoundarySemantic::Door
        );
        assert_eq!(
            BoundarySegment::infer_semantic_from_layer("入户门"),
            BoundarySemantic::Door
        );
        assert_eq!(
            BoundarySegment::infer_semantic_from_layer("室内门"),
            BoundarySemantic::Door
        );
        assert_eq!(
            BoundarySegment::infer_semantic_from_layer("防火门"),
            BoundarySemantic::Door
        );
        assert_eq!(
            BoundarySegment::infer_semantic_from_layer("单开门"),
            BoundarySemantic::Door
        );
        assert_eq!(
            BoundarySegment::infer_semantic_from_layer("双开门"),
            BoundarySemantic::Door
        );
        assert_eq!(
            BoundarySegment::infer_semantic_from_layer("推拉门"),
            BoundarySemantic::Door
        );

        // AIA 标准
        assert_eq!(
            BoundarySegment::infer_semantic_from_layer("A-DOOR"),
            BoundarySemantic::Door
        );
        assert_eq!(
            BoundarySegment::infer_semantic_from_layer("A-DOOR-EXT"),
            BoundarySemantic::Door
        );

        // 模式匹配
        assert_eq!(
            BoundarySegment::infer_semantic_from_layer("DOOR-INT"),
            BoundarySemantic::Door
        );
        assert_eq!(
            BoundarySegment::infer_semantic_from_layer("DOOR_01"),
            BoundarySemantic::Door
        );
        assert_eq!(
            BoundarySegment::infer_semantic_from_layer("D-01"),
            BoundarySemantic::Door
        );
    }

    #[test]
    fn test_infer_semantic_window() {
        // 窗关键词
        assert_eq!(
            BoundarySegment::infer_semantic_from_layer("WINDOW"),
            BoundarySemantic::Window
        );
        assert_eq!(
            BoundarySegment::infer_semantic_from_layer("采光窗"),
            BoundarySemantic::Window
        );
        assert_eq!(
            BoundarySegment::infer_semantic_from_layer("天窗"),
            BoundarySemantic::Window
        );
        assert_eq!(
            BoundarySegment::infer_semantic_from_layer("落地窗"),
            BoundarySemantic::Window
        );
        assert_eq!(
            BoundarySegment::infer_semantic_from_layer("百叶窗"),
            BoundarySemantic::Window
        );

        // AIA 标准
        assert_eq!(
            BoundarySegment::infer_semantic_from_layer("A-WIND"),
            BoundarySemantic::Window
        );
        assert_eq!(
            BoundarySegment::infer_semantic_from_layer("A-WIND-EXT"),
            BoundarySemantic::Window
        );

        // 模式匹配
        assert_eq!(
            BoundarySegment::infer_semantic_from_layer("WINDOW-01"),
            BoundarySemantic::Window
        );
        assert_eq!(
            BoundarySegment::infer_semantic_from_layer("GLASS-FRONT"),
            BoundarySemantic::Window
        );
        assert_eq!(
            BoundarySegment::infer_semantic_from_layer("W-01"),
            BoundarySemantic::Window
        );
    }

    #[test]
    fn test_infer_semantic_opening() {
        // 开口关键词
        assert_eq!(
            BoundarySegment::infer_semantic_from_layer("OPEN"),
            BoundarySemantic::Opening
        );
        assert_eq!(
            BoundarySegment::infer_semantic_from_layer("开口"),
            BoundarySemantic::Opening
        );
        assert_eq!(
            BoundarySegment::infer_semantic_from_layer("OPENING"),
            BoundarySemantic::Opening
        );
        assert_eq!(
            BoundarySegment::infer_semantic_from_layer("HOLE"),
            BoundarySemantic::Opening
        );
        assert_eq!(
            BoundarySegment::infer_semantic_from_layer("通道"),
            BoundarySemantic::Opening
        );

        // AIA 标准
        assert_eq!(
            BoundarySegment::infer_semantic_from_layer("A-OPEN"),
            BoundarySemantic::Opening
        );
        assert_eq!(
            BoundarySegment::infer_semantic_from_layer("OPENING-01"),
            BoundarySemantic::Opening
        );
    }

    #[test]
    fn test_infer_semantic_wall() {
        // 墙关键词
        assert_eq!(
            BoundarySegment::infer_semantic_from_layer("WALL"),
            BoundarySemantic::HardWall
        );
        assert_eq!(
            BoundarySegment::infer_semantic_from_layer("墙体"),
            BoundarySemantic::HardWall
        );
        assert_eq!(
            BoundarySegment::infer_semantic_from_layer("内墙"),
            BoundarySemantic::HardWall
        );
        assert_eq!(
            BoundarySegment::infer_semantic_from_layer("外墙"),
            BoundarySemantic::HardWall
        );
        assert_eq!(
            BoundarySegment::infer_semantic_from_layer("剪力墙"),
            BoundarySemantic::HardWall
        );

        // 结构
        assert_eq!(
            BoundarySegment::infer_semantic_from_layer("STRUCT"),
            BoundarySemantic::HardWall
        );
        assert_eq!(
            BoundarySegment::infer_semantic_from_layer("COLUMN"),
            BoundarySemantic::HardWall
        );
        assert_eq!(
            BoundarySegment::infer_semantic_from_layer("BEAM"),
            BoundarySemantic::HardWall
        );

        // AIA 标准
        assert_eq!(
            BoundarySegment::infer_semantic_from_layer("A-WALL"),
            BoundarySemantic::HardWall
        );
        assert_eq!(
            BoundarySegment::infer_semantic_from_layer("S-WALL"),
            BoundarySemantic::HardWall
        );
        assert_eq!(
            BoundarySegment::infer_semantic_from_layer("S-STRC"),
            BoundarySemantic::HardWall
        );
    }

    #[test]
    fn test_infer_semantic_default() {
        // 未知图层默认为硬墙
        assert_eq!(
            BoundarySegment::infer_semantic_from_layer("UNKNOWN"),
            BoundarySemantic::HardWall
        );
        assert_eq!(
            BoundarySegment::infer_semantic_from_layer("0"),
            BoundarySemantic::HardWall
        );
        assert_eq!(
            BoundarySegment::infer_semantic_from_layer("FURNITURE"),
            BoundarySemantic::HardWall
        );
    }

    #[test]
    fn test_infer_material_from_color() {
        // ACI 颜色映射
        assert_eq!(
            BoundarySegment::infer_material_from_aci_color(1),
            Some("concrete".to_string())
        );
        assert_eq!(
            BoundarySegment::infer_material_from_aci_color(2),
            Some("brick".to_string())
        );
        assert_eq!(
            BoundarySegment::infer_material_from_aci_color(3),
            Some("wood".to_string())
        );
        assert_eq!(
            BoundarySegment::infer_material_from_aci_color(4),
            Some("gypsum".to_string())
        );
        assert_eq!(
            BoundarySegment::infer_material_from_aci_color(5),
            Some("glass".to_string())
        );
        assert_eq!(
            BoundarySegment::infer_material_from_aci_color(6),
            Some("metal".to_string())
        );
        assert_eq!(
            BoundarySegment::infer_material_from_aci_color(7),
            Some("default_wall".to_string())
        );

        // 未知颜色
        assert_eq!(BoundarySegment::infer_material_from_aci_color(0), None);
        assert_eq!(BoundarySegment::infer_material_from_aci_color(8), None);
        assert_eq!(BoundarySegment::infer_material_from_aci_color(255), None);
    }

    #[test]
    fn test_infer_material_from_layer() {
        // 混凝土
        assert_eq!(
            BoundarySegment::infer_material_from_layer("WALL-CONC"),
            Some("concrete".to_string())
        );
        assert_eq!(
            BoundarySegment::infer_material_from_layer("混凝土墙"),
            Some("concrete".to_string())
        );

        // 砖墙
        assert_eq!(
            BoundarySegment::infer_material_from_layer("BRICK"),
            Some("brick".to_string())
        );
        assert_eq!(
            BoundarySegment::infer_material_from_layer("砖墙"),
            Some("brick".to_string())
        );

        // 木材
        assert_eq!(
            BoundarySegment::infer_material_from_layer("WOOD"),
            Some("wood".to_string())
        );
        assert_eq!(
            BoundarySegment::infer_material_from_layer("木地板"),
            Some("wood".to_string())
        );

        // 玻璃
        assert_eq!(
            BoundarySegment::infer_material_from_layer("GLASS"),
            Some("glass".to_string())
        );
        assert_eq!(
            BoundarySegment::infer_material_from_layer("玻璃幕墙"),
            Some("glass".to_string())
        );

        // 金属
        assert_eq!(
            BoundarySegment::infer_material_from_layer("METAL"),
            Some("metal".to_string())
        );
        assert_eq!(
            BoundarySegment::infer_material_from_layer("STEEL"),
            Some("metal".to_string())
        );
        assert_eq!(
            BoundarySegment::infer_material_from_layer("钢结构"),
            Some("metal".to_string())
        );

        // 石膏板
        assert_eq!(
            BoundarySegment::infer_material_from_layer("GYPSUM"),
            Some("gypsum".to_string())
        );
        assert_eq!(
            BoundarySegment::infer_material_from_layer("石膏板"),
            Some("gypsum".to_string())
        );

        // 未知材料
        assert_eq!(BoundarySegment::infer_material_from_layer("UNKNOWN"), None);
    }

    #[test]
    fn test_calculate_width() {
        // 标准门宽度 (900mm = 0.9m)
        let start = [0.0, 0.0];
        let end = [900.0, 0.0];
        let width = BoundarySegment::calculate_width(start, end);
        assert!(width.is_some());
        assert!((width.unwrap() - 0.9).abs() < 0.01);

        // 标准窗宽度 (1500mm = 1.5m)
        let start = [0.0, 0.0];
        let end = [1500.0, 0.0];
        let width = BoundarySegment::calculate_width(start, end);
        assert!(width.is_some());
        assert!((width.unwrap() - 1.5).abs() < 0.01);

        // 太窄 (< 0.5m)
        let start = [0.0, 0.0];
        let end = [300.0, 0.0];
        let width = BoundarySegment::calculate_width(start, end);
        assert!(width.is_none());

        // 太宽 (> 5.0m)
        let start = [0.0, 0.0];
        let end = [6000.0, 0.0];
        let width = BoundarySegment::calculate_width(start, end);
        assert!(width.is_none());

        // 斜向开口
        let start = [0.0, 0.0];
        let end = [3000.0, 4000.0];
        let width = BoundarySegment::calculate_width(start, end);
        assert!(width.is_some());
        assert!((width.unwrap() - 5.0).abs() < 0.01);
    }

    #[test]
    fn test_infer_from_layer() {
        // 门
        let segment =
            BoundarySegment::infer_from_layer("DOOR-01", Some("wood".to_string()), Some(0.9));
        assert_eq!(segment.semantic, BoundarySemantic::Door);
        assert_eq!(segment.material, Some("wood".to_string()));
        assert_eq!(segment.width, Some(0.9));

        // 窗
        let segment =
            BoundarySegment::infer_from_layer("WINDOW-01", Some("glass".to_string()), Some(1.5));
        assert_eq!(segment.semantic, BoundarySemantic::Window);
        assert_eq!(segment.material, Some("glass".to_string()));
        assert_eq!(segment.width, Some(1.5));

        // 墙
        let segment =
            BoundarySegment::infer_from_layer("WALL-EXT", Some("concrete".to_string()), None);
        assert_eq!(segment.semantic, BoundarySemantic::HardWall);
        assert_eq!(segment.material, Some("concrete".to_string()));
        assert_eq!(segment.width, None);
    }
}
