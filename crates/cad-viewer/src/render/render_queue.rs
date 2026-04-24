//! 渲染队列（批量合并优化 - P1-1 修复：HashMap O(1) 查找）

use eframe::egui;
use egui::{Color32, Pos2, Stroke};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

/// 渲染队列（批量合并 - P1-1 优化：使用 HashMap 实现 O(1) 查找）
pub struct RenderQueue {
    batches: Vec<RenderBatch>,
    /// P1-1 优化：批量索引，key = (color, line_width, layer_name)
    batch_index: HashMap<BatchKey, usize>,
}

/// 渲染批次（按材质/图层分组）
#[allow(dead_code)]
pub struct RenderBatch {
    pub material: MaterialId,
    pub layer: LayerId,
    pub segments: Vec<(Pos2, Pos2)>,
    pub stroke: Stroke,
}

/// 材质 ID
#[derive(Debug, Clone, PartialEq)]
pub struct MaterialId {
    pub color: Color32,
    pub line_width: f32,
}

/// 图层 ID
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct LayerId {
    pub name: String,
}

/// P1-1 优化：批量查找键（包装 f32 以支持 Hash 和 Eq）
#[derive(Debug, Clone)]
struct BatchKey {
    color: Color32,
    line_width_quantized: u8, // P11 修复：量化后的线宽（0.1px 精度）
    layer_name: String,
}

/// P11 修复：量化线宽到离散级别
/// 将线宽量化到 0.1px 精度，避免 0.999 和 1.000 被视为不同批次
/// 例如：0.5-0.54 -> 5, 0.55-0.64 -> 6, etc.
fn quantize_line_width(width: f32) -> u8 {
    (width * 10.0).round() as u8
}

impl BatchKey {
    fn new(color: Color32, line_width: f32, layer_name: String) -> Self {
        Self {
            color,
            line_width_quantized: quantize_line_width(line_width), // P11 修复：量化线宽
            layer_name,
        }
    }
}

impl PartialEq for BatchKey {
    fn eq(&self, other: &Self) -> bool {
        self.color == other.color
            && self.line_width_quantized == other.line_width_quantized
            && self.layer_name == other.layer_name
    }
}

impl Eq for BatchKey {}

impl Hash for BatchKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.color.hash(state);
        self.line_width_quantized.hash(state);
        self.layer_name.hash(state);
    }
}

impl Default for RenderQueue {
    fn default() -> Self {
        Self::new()
    }
}

impl RenderQueue {
    pub fn new() -> Self {
        Self {
            batches: Vec::new(),
            batch_index: HashMap::new(),
        }
    }

    /// P1-1 优化：添加线段到队列（O(1) 查找）
    pub fn add_line(&mut self, start: Pos2, end: Pos2, material: MaterialId, layer: LayerId) {
        // 创建查找键
        let key = BatchKey::new(material.color, material.line_width, layer.name.clone());

        // O(1) 查找现有批次
        if let Some(&batch_idx) = self.batch_index.get(&key) {
            // 找到现有批次，添加线段
            self.batches[batch_idx].segments.push((start, end));
        } else {
            // 创建新批次
            let batch_idx = self.batches.len();
            self.batches.push(RenderBatch {
                material: material.clone(),
                layer: layer.clone(),
                segments: vec![(start, end)],
                stroke: Stroke::new(material.line_width, material.color),
            });
            // 更新索引
            self.batch_index.insert(key, batch_idx);
        }
    }

    /// 清空队列
    pub fn clear(&mut self) {
        self.batches.clear();
        self.batch_index.clear();
    }

    /// 渲染所有批次
    pub fn render(&self, painter: &egui::Painter) {
        for batch in &self.batches {
            for &(start, end) in &batch.segments {
                painter.line_segment([start, end], batch.stroke);
            }
        }
    }

    /// 获取批次数量（用于性能分析）
    #[allow(dead_code)]
    pub fn batch_count(&self) -> usize {
        self.batches.len()
    }

    /// 获取总线段数（用于性能分析）
    #[allow(dead_code)]
    pub fn total_segments(&self) -> usize {
        self.batches.iter().map(|b| b.segments.len()).sum()
    }

    /// P1-1 新增：获取索引大小（用于性能分析）
    #[allow(dead_code)]
    pub fn index_size(&self) -> usize {
        self.batch_index.len()
    }

    /// P1-1 新增：计算批量合并率（用于性能分析）
    #[allow(dead_code)]
    pub fn batch_efficiency(&self) -> f64 {
        let total_segments = self.total_segments();
        if total_segments == 0 {
            return 0.0;
        }
        // 合并率 = 平均每个批次的线段数 / 总线段数

        total_segments as f64 / self.batch_count() as f64
    }
}
