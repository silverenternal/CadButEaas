//! 导出格式定义

use serde::{Serialize, Deserialize};
use common_types::{SceneState, SourceType};

/// 导出格式枚举
#[derive(Debug, Clone, Copy)]
pub enum ExportFormat {
    Json,
    Binary,
}

/// JSON Schema 版本
pub const SCHEMA_VERSION: &str = "1.0";

/// 导出的场景 JSON 结构
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SceneJson {
    /// Schema 版本
    pub schema_version: &'static str,
    /// 单位
    pub units: String,
    /// 坐标系
    pub coordinate_system: String,
    /// 几何数据
    pub geometry: GeometryData,
    /// 边界标注
    #[serde(default)]
    pub boundaries: Vec<BoundaryData>,
    /// 声源 (可选)
    #[serde(default)]
    pub sources: Vec<SourceData>,
}

/// 几何数据
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeometryData {
    /// 外轮廓
    pub outer: Vec<[f64; 2]>,
    /// 孔洞列表
    #[serde(default)]
    pub holes: Vec<Vec<[f64; 2]>>,
}

/// 边界数据
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundaryData {
    /// 线段索引
    pub segment: [usize; 2],
    /// 语义类型
    pub semantic: String,
    /// 材料 (可选)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub material: Option<String>,
    /// 宽度 (仅用于开口)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub width: Option<f64>,
}

/// 声源数据
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceData {
    pub id: String,
    pub position: [f64; 3],
    /// 朝向（可选）
    #[serde(skip_serializing_if = "Option::is_none")]
    pub orientation: Option<[f64; 2]>,
    /// 指向性类型
    #[serde(default)]
    pub source_type: SourceType,
    /// 增益 (dB)
    #[serde(default)]
    pub gain_db: f64,
    /// 延迟 (ms)
    #[serde(default)]
    pub delay_ms: f64,
}

impl SceneJson {
    /// 从 SceneState 转换
    pub fn from_scene_state(scene: &SceneState) -> Self {
        let outer = scene.outer.as_ref().map(|l| l.points.clone()).unwrap_or_default();
        
        let holes: Vec<Vec<[f64; 2]>> = scene.holes.iter().map(|h| h.points.clone()).collect();
        
        let boundaries: Vec<BoundaryData> = scene.boundaries.iter().map(|b| {
            BoundaryData {
                segment: b.segment,
                semantic: format!("{:?}", b.semantic),
                material: b.material.clone(),
                width: b.width,
            }
        }).collect();
        
        let sources: Vec<SourceData> = scene.sources.iter().map(|s| {
            SourceData {
                id: s.id.clone(),
                position: s.position,
                orientation: s.orientation,
                source_type: s.source_type.clone(),
                gain_db: s.gain_db,
                delay_ms: s.delay_ms,
            }
        }).collect();

        Self {
            schema_version: SCHEMA_VERSION,
            units: format!("{:?}", scene.units),
            coordinate_system: format!("{:?}", scene.coordinate_system),
            geometry: GeometryData { outer, holes },
            boundaries,
            sources,
        }
    }

    /// 转换为字节 (JSON)
    pub fn to_json_bytes(&self, pretty: bool) -> Result<Vec<u8>, serde_json::Error> {
        if pretty {
            serde_json::to_vec_pretty(self)
        } else {
            serde_json::to_vec(self)
        }
    }

    /// 转换为二进制 (bincode)
    pub fn to_binary_bytes(&self) -> Result<Vec<u8>, bincode::Error> {
        bincode::serialize(self)
    }
}
