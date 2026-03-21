//! 场景状态（业务数据）
//!
//! 包含所有场景相关的业务数据：边、座椅区域、图层等

use interact::Edge;
use common_types::{SeatZone, RenderConfig};
use std::collections::HashMap;
use std::path::PathBuf;

/// 场景状态（业务数据）
pub struct SceneState {
    /// 所有边
    pub edges: Vec<Edge>,
    /// 座椅区域列表
    pub seat_zones: Vec<SeatZone>,
    /// 图层集合
    pub layers: LayerCollection,
    /// 文件路径
    pub file_path: Option<PathBuf>,
    /// 场景原点（用于相对坐标渲染，提升大坐标场景精度）
    pub scene_origin: [f64; 2],
    /// 渲染配置（LOD 设置）
    pub render_config: Option<RenderConfig>,
    /// 统计信息
    pub stats: SceneStats,
}

/// 图层集合（管理图层可见性）
pub struct LayerCollection {
    /// 图层名 -> 是否可见
    pub visibility: HashMap<String, bool>,
    /// 当前过滤模式
    pub filter_mode: String,
}

impl LayerCollection {
    /// 获取图层可见性
    pub fn is_visible(&self, layer: &str) -> bool {
        *self.visibility.get(layer).unwrap_or(&true)
    }
}

/// 场景统计信息
pub struct SceneStats {
    /// 总边数
    pub total_edges: usize,
    /// 可见边数
    pub visible_edges: usize,
}

impl Default for SceneState {
    fn default() -> Self {
        Self {
            edges: Vec::new(),
            seat_zones: Vec::new(),
            layers: LayerCollection::default(),
            file_path: None,
            scene_origin: [0.0, 0.0],
            render_config: None,
            stats: SceneStats::default(),
        }
    }
}

impl Default for LayerCollection {
    fn default() -> Self {
        Self {
            visibility: HashMap::new(),
            filter_mode: "All".to_string(),
        }
    }
}

impl Default for SceneStats {
    fn default() -> Self {
        Self {
            total_edges: 0,
            visible_edges: 0,
        }
    }
}

impl SceneState {
    /// 创建新的场景状态
    pub fn new() -> Self {
        Self::default()
    }

    /// 更新图层可见性统计
    pub fn update_visibility_stats(&mut self) {
        self.stats.total_edges = self.edges.len();
        self.stats.visible_edges = self.edges.iter().filter(|e| {
            if let Some(visible) = e.visible {
                visible
            } else if let Some(layer) = &e.layer {
                *self.layers.visibility.get(layer).unwrap_or(&true)
            } else {
                true
            }
        }).count();
    }

    /// 切换图层可见性（用于图层面板交互）
    #[allow(dead_code)]
    pub fn toggle_layer(&mut self, layer: &str) {
        let visible = self.layers.visibility.get(layer).copied().unwrap_or(true);
        self.layers.visibility.insert(layer.to_string(), !visible);

        // 更新边的可见性
        for edge in &mut self.edges {
            if edge.layer.as_deref() == Some(layer) {
                edge.visible = Some(!visible);
            }
        }

        self.update_visibility_stats();
    }

    /// 设置图层过滤模式
    pub fn set_layer_filter(&mut self, mode: &str) {
        self.layers.filter_mode = mode.to_string();
        self.apply_filter(mode);
    }

    /// 应用图层过滤
    fn apply_filter(&mut self, mode: &str) {
        // 重置所有图层为可见
        self.layers.visibility.clear();

        for edge in &mut self.edges {
            edge.visible = None; // 重置
        }

        match mode {
            "All" => {
                // 全部可见
            }
            "Walls" => {
                for edge in &mut self.edges {
                    let layer = edge.layer.as_deref().unwrap_or("");
                    let is_wall = layer.to_uppercase().contains("WALL")
                        || layer.to_uppercase().contains("墙")
                        || layer.to_uppercase().contains("STRUCT");
                    if !is_wall {
                        edge.visible = Some(false);
                        self.layers.visibility.insert(layer.to_string(), false);
                    } else {
                        self.layers.visibility.insert(layer.to_string(), true);
                    }
                }
            }
            "Openings" => {
                for edge in &mut self.edges {
                    let layer = edge.layer.as_deref().unwrap_or("");
                    let upper = layer.to_uppercase();
                    let is_opening = upper.contains("DOOR") || upper.contains("门")
                        || upper.contains("WINDOW") || upper.contains("窗")
                        || upper.contains("OPEN");
                    if !is_opening {
                        edge.visible = Some(false);
                        self.layers.visibility.insert(layer.to_string(), false);
                    } else {
                        self.layers.visibility.insert(layer.to_string(), true);
                    }
                }
            }
            "Architectural" => {
                for edge in &mut self.edges {
                    let layer = edge.layer.as_deref().unwrap_or("");
                    let upper = layer.to_uppercase();
                    let is_arch = upper.contains("WALL") || upper.contains("墙")
                        || upper.contains("DOOR") || upper.contains("门")
                        || upper.contains("WINDOW") || upper.contains("窗")
                        || upper.contains("STRUCT");
                    if !is_arch {
                        edge.visible = Some(false);
                        self.layers.visibility.insert(layer.to_string(), false);
                    } else {
                        self.layers.visibility.insert(layer.to_string(), true);
                    }
                }
            }
            "Furniture" => {
                for edge in &mut self.edges {
                    let layer = edge.layer.as_deref().unwrap_or("");
                    let is_furniture = layer.to_uppercase().contains("FURN")
                        || layer.to_uppercase().contains("家具");
                    if !is_furniture {
                        edge.visible = Some(false);
                        self.layers.visibility.insert(layer.to_string(), false);
                    } else {
                        self.layers.visibility.insert(layer.to_string(), true);
                    }
                }
            }
            _ => {}
        }

        self.update_visibility_stats();
    }

    /// 获取所有唯一图层名称
    pub fn get_unique_layers(&self) -> Vec<String> {
        let mut layers: Vec<String> = self.edges.iter()
            .filter_map(|e| e.layer.clone())
            .collect();
        layers.sort();
        layers.dedup();
        layers
    }

    /// 获取图层可见性
    pub fn is_layer_visible(&self, layer: &str) -> bool {
        self.layers.is_visible(layer)
    }

    /// 计算场景边界
    pub fn calculate_bounds(&self) -> Option<([f64; 2], [f64; 2])> {
        if self.edges.is_empty() {
            return None;
        }

        let mut min_x = f64::INFINITY;
        let mut max_x = f64::NEG_INFINITY;
        let mut min_y = f64::INFINITY;
        let mut max_y = f64::NEG_INFINITY;

        for edge in &self.edges {
            min_x = min_x.min(edge.start[0]).min(edge.end[0]);
            max_x = max_x.max(edge.start[0]).max(edge.end[0]);
            min_y = min_y.min(edge.start[1]).min(edge.end[1]);
            max_y = max_y.max(edge.start[1]).max(edge.end[1]);
        }

        // 添加 10% 边距
        let margin_x = (max_x - min_x) * 0.1;
        let margin_y = (max_y - min_y) * 0.1;

        Some((
            [min_x - margin_x, min_y - margin_y],
            [max_x + margin_x, max_y + margin_y],
        ))
    }
}
