//! 渲染队列（批量合并优化）

use eframe::egui;
use egui::{Color32, Pos2, Stroke};

/// 渲染队列（批量合并）
pub struct RenderQueue {
    batches: Vec<RenderBatch>,
}

/// 渲染批次（按材质/图层分组）
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

impl Default for RenderQueue {
    fn default() -> Self {
        Self::new()
    }
}

impl RenderQueue {
    pub fn new() -> Self {
        Self {
            batches: Vec::new(),
        }
    }

    /// 添加线段到队列
    pub fn add_line(&mut self, start: Pos2, end: Pos2, material: MaterialId, layer: LayerId) {
        // 查找现有批次
        if let Some(batch) = self.batches.iter_mut().find(|b| {
            b.material.color == material.color && b.material.line_width == material.line_width && b.layer == layer
        }) {
            batch.segments.push((start, end));
        } else {
            // 创建新批次
            self.batches.push(RenderBatch {
                material: material.clone(),
                layer,
                segments: vec![(start, end)],
                stroke: Stroke::new(material.line_width, material.color),
            });
        }
    }

    /// 清空队列
    pub fn clear(&mut self) {
        self.batches.clear();
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
}
