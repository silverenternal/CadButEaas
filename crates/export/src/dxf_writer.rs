//! DXF 文件导出器
//!
//! 将解析后的实体导出为 DXF 文件，用于可视化验证
//!
//! ## 功能
//! - 基本实体导出（Line, Polyline, Arc, Circle）
//! - 块引用导出（BlockReference）

use common_types::{Point2, RawEntity};
use dxf::Block;
use dxf::{
    entities::Arc, entities::Circle, entities::Entity, entities::EntityType, entities::Line,
    entities::LwPolyline, Drawing,
};
use std::path::Path;

/// DXF 导出器
pub struct DxfWriter {
    drawing: Drawing,
    /// 已定义的块名称集合（用于避免重复定义）
    defined_blocks: std::collections::HashSet<String>,
}

impl DxfWriter {
    /// 创建新的 DXF 导出器
    pub fn new() -> Self {
        Self {
            drawing: Drawing::new(),
            defined_blocks: std::collections::HashSet::new(),
        }
    }

    /// 添加直线
    pub fn add_line(&mut self, start: Point2, end: Point2, layer: &str) {
        let line = EntityType::Line(Line {
            p1: dxf::Point::new(start[0], start[1], 0.0),
            p2: dxf::Point::new(end[0], end[1], 0.0),
            ..Default::default()
        });

        let mut entity = Entity::new(line);
        entity.common.layer = layer.to_string();
        self.drawing.add_entity(entity);
    }

    /// 添加多段线
    pub fn add_polyline(&mut self, points: &[Point2], closed: bool, layer: &str) {
        if points.is_empty() {
            return;
        }

        let vertices: Vec<_> = points
            .iter()
            .map(|p| dxf::LwPolylineVertex {
                x: p[0],
                y: p[1],
                ..Default::default()
            })
            .collect();

        // 1 = CLOSED flag for LwPolyline
        let flags = if closed { 1 } else { 0 };

        let polyline = EntityType::LwPolyline(LwPolyline {
            vertices,
            flags,
            ..Default::default()
        });

        let mut entity = Entity::new(polyline);
        entity.common.layer = layer.to_string();
        self.drawing.add_entity(entity);
    }

    /// 添加圆弧
    pub fn add_arc(
        &mut self,
        center: Point2,
        radius: f64,
        start_angle: f64,
        end_angle: f64,
        layer: &str,
    ) {
        let arc = EntityType::Arc(Arc {
            center: dxf::Point::new(center[0], center[1], 0.0),
            radius,
            start_angle: start_angle.to_radians(),
            end_angle: end_angle.to_radians(),
            ..Default::default()
        });

        let mut entity = Entity::new(arc);
        entity.common.layer = layer.to_string();
        self.drawing.add_entity(entity);
    }

    /// 添加圆
    pub fn add_circle(&mut self, center: Point2, radius: f64, layer: &str) {
        let circle = EntityType::Circle(Circle {
            center: dxf::Point::new(center[0], center[1], 0.0),
            radius,
            ..Default::default()
        });

        let mut entity = Entity::new(circle);
        entity.common.layer = layer.to_string();
        self.drawing.add_entity(entity);
    }

    /// 添加块定义（BLOCK + ENDBLK）
    ///
    /// ## 参数
    /// - `name`: 块名称
    /// - `entities`: 块内实体列表
    /// - `base_point`: 块基点（插入时的参考点）
    ///
    /// ## DXF 结构
    /// ```text
    /// BLOCK
    ///   2: 块名
    ///   70: 块标志（1=匿名，2=非一致缩放，4=外部引用，8=布局，16=3D 缩放）
    ///  10/20/30: 基点
    ///   1: 外部参照文件名（如果有）
    ///   ... 块内实体 ...
    /// ENDBLK
    /// ```
    pub fn add_block_definition(
        &mut self,
        name: &str,
        _entities: &[RawEntity],
        base_point: Point2,
    ) {
        // 避免重复定义
        if self.defined_blocks.contains(name) {
            return;
        }

        // 创建 BLOCK 实体
        let block = Block {
            name: name.to_string(),
            base_point: dxf::Point::new(base_point[0], base_point[1], 0.0),
            ..Default::default()
        };

        // 添加到绘图的块定义表
        self.drawing.add_block(block);
        self.defined_blocks.insert(name.to_string());

        // 注意：dxf 0.6.0 的块定义实体管理需要更复杂的处理
        // 当前实现仅创建块定义头信息，块内实体通过 add_entities 添加到主绘图
        // 完整的块定义支持（包含块内实体）需要扩展 dxf crate 或使用其内部 API
    }

    /// 添加块引用（INSERT 实体）
    pub fn add_block_reference(
        &mut self,
        name: &str,
        insertion_point: Point2,
        scale: [f64; 3],
        rotation_deg: f64,
        layer: &str,
    ) {
        use dxf::entities::Insert;

        let insert = EntityType::Insert(Insert {
            name: name.to_string(),
            location: dxf::Point::new(insertion_point[0], insertion_point[1], 0.0),
            x_scale_factor: scale[0],
            y_scale_factor: scale[1],
            z_scale_factor: scale[2],
            rotation: rotation_deg,
            ..Default::default()
        });

        let mut entity = Entity::new(insert);
        entity.common.layer = layer.to_string();
        self.drawing.add_entity(entity);
    }

    /// 从实体列表批量添加
    pub fn add_entities(&mut self, entities: &[RawEntity]) {
        for entity in entities {
            match entity {
                RawEntity::Line {
                    start,
                    end,
                    metadata,
                    ..
                } => {
                    let layer = metadata.layer.as_deref().unwrap_or("0");
                    self.add_line(*start, *end, layer);
                }
                RawEntity::Polyline {
                    points,
                    closed,
                    metadata,
                    ..
                } => {
                    let layer = metadata.layer.as_deref().unwrap_or("0");
                    self.add_polyline(points, *closed, layer);
                }
                RawEntity::Arc {
                    center,
                    radius,
                    start_angle,
                    end_angle,
                    metadata,
                    ..
                } => {
                    let layer = metadata.layer.as_deref().unwrap_or("0");
                    self.add_arc(*center, *radius, *start_angle, *end_angle, layer);
                }
                RawEntity::Circle {
                    center,
                    radius,
                    metadata,
                    ..
                } => {
                    let layer = metadata.layer.as_deref().unwrap_or("0");
                    self.add_circle(*center, *radius, layer);
                }
                RawEntity::BlockReference {
                    block_name,
                    insertion_point,
                    scale,
                    rotation,
                    metadata,
                    ..
                } => {
                    let layer = metadata.layer.as_deref().unwrap_or("0");
                    self.add_block_reference(
                        block_name,
                        *insertion_point,
                        *scale,
                        *rotation,
                        layer,
                    );
                }
                _ => {} // 其他实体类型暂不支持
            }
        }
    }

    /// 保存 DXF 文件
    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), String> {
        let path = path.as_ref();
        self.drawing
            .save_file(path)
            .map_err(|e| format!("保存 DXF 文件失败：{}", e))
    }

    /// 获取绘图对象引用
    pub fn drawing(&self) -> &Drawing {
        &self.drawing
    }

    /// 获取块定义列表
    pub fn block_definitions(&self) -> impl Iterator<Item = &Block> {
        self.drawing.blocks()
    }
}

impl Default for DxfWriter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn count_entities(drawing: &Drawing) -> usize {
        drawing.entities().count()
    }

    #[test]
    fn test_dxf_writer_creation() {
        let writer = DxfWriter::new();
        assert_eq!(count_entities(&writer.drawing), 0);
    }

    #[test]
    fn test_add_line() {
        let mut writer = DxfWriter::new();
        writer.add_line([0.0, 0.0], [10.0, 10.0], "WALL");
        assert_eq!(count_entities(&writer.drawing), 1);
    }

    #[test]
    fn test_add_polyline() {
        let mut writer = DxfWriter::new();
        let points = vec![[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]];
        writer.add_polyline(&points, true, "WALL");
        assert_eq!(count_entities(&writer.drawing), 1);
    }

    #[test]
    fn test_add_arc() {
        let mut writer = DxfWriter::new();
        writer.add_arc([0.0, 0.0], 5.0, 0.0, 90.0, "ARC");
        assert_eq!(count_entities(&writer.drawing), 1);
    }

    #[test]
    fn test_add_circle() {
        let mut writer = DxfWriter::new();
        writer.add_circle([0.0, 0.0], 5.0, "CIRCLE");
        assert_eq!(count_entities(&writer.drawing), 1);
    }

    #[test]
    fn test_add_entities() {
        let mut writer = DxfWriter::new();
        let entities = vec![
            RawEntity::Line {
                start: [0.0, 0.0],
                end: [10.0, 0.0],
                metadata: common_types::EntityMetadata {
                    layer: Some("WALL".to_string()),
                    ..Default::default()
                },
                semantic: None,
            },
            RawEntity::Polyline {
                points: vec![[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]],
                closed: true,
                metadata: common_types::EntityMetadata {
                    layer: Some("ROOM".to_string()),
                    ..Default::default()
                },
                semantic: None,
            },
        ];
        writer.add_entities(&entities);
        assert_eq!(count_entities(&writer.drawing), 2);
    }

    #[test]
    fn test_save_and_load() {
        let temp_path = std::env::temp_dir().join("test_dxf_writer.dxf");

        // 创建并保存
        let mut writer = DxfWriter::new();
        writer.add_line([0.0, 0.0], [10.0, 10.0], "TEST");
        writer.save(&temp_path).unwrap();

        // 验证文件存在
        assert!(temp_path.exists());

        // 清理
        let _ = fs::remove_file(temp_path);
    }

    #[test]
    fn test_add_block_reference() {
        let mut writer = DxfWriter::new();

        // 添加块引用（不需要预先定义块）
        writer.add_block_reference("TEST_BLOCK", [100.0, 100.0], [1.0, 1.0, 1.0], 45.0, "0");

        // 验证引用存在
        let insert_count = writer
            .drawing()
            .entities()
            .filter(|e| matches!(e.specific, EntityType::Insert(_)))
            .count();
        assert_eq!(insert_count, 1, "应该有 1 个块引用");
    }

    #[test]
    fn test_add_block_definition() {
        let mut writer = DxfWriter::new();

        // 创建块内实体
        let block_entities = vec![
            RawEntity::Line {
                start: [0.0, 0.0],
                end: [10.0, 0.0],
                metadata: common_types::EntityMetadata {
                    layer: Some("0".to_string()),
                    ..Default::default()
                },
                semantic: None,
            },
            RawEntity::Line {
                start: [10.0, 0.0],
                end: [10.0, 10.0],
                metadata: common_types::EntityMetadata {
                    layer: Some("0".to_string()),
                    ..Default::default()
                },
                semantic: None,
            },
            RawEntity::Line {
                start: [10.0, 10.0],
                end: [0.0, 10.0],
                metadata: common_types::EntityMetadata {
                    layer: Some("0".to_string()),
                    ..Default::default()
                },
                semantic: None,
            },
            RawEntity::Line {
                start: [0.0, 10.0],
                end: [0.0, 0.0],
                metadata: common_types::EntityMetadata {
                    layer: Some("0".to_string()),
                    ..Default::default()
                },
                semantic: None,
            },
        ];

        // 添加块定义
        writer.add_block_definition("TEST_BLOCK_DEF", &block_entities, [0.0, 0.0]);

        // 验证块定义存在
        let block_count = writer.block_definitions().count();
        assert_eq!(block_count, 1, "应该有 1 个块定义");

        let block = writer.block_definitions().next().unwrap();
        assert_eq!(block.name, "TEST_BLOCK_DEF");

        // 测试重复定义（应该被跳过）
        writer.add_block_definition("TEST_BLOCK_DEF", &block_entities, [0.0, 0.0]);
        assert_eq!(writer.block_definitions().count(), 1, "重复定义应该被跳过");
    }

    #[test]
    fn test_block_definition_and_reference() {
        let mut writer = DxfWriter::new();

        // 创建块定义
        let block_entities = vec![RawEntity::Line {
            start: [0.0, 0.0],
            end: [10.0, 0.0],
            metadata: common_types::EntityMetadata {
                layer: Some("0".to_string()),
                ..Default::default()
            },
            semantic: None,
        }];
        writer.add_block_definition("MY_BLOCK", &block_entities, [0.0, 0.0]);

        // 添加块引用
        writer.add_block_reference("MY_BLOCK", [100.0, 100.0], [1.0, 1.0, 1.0], 0.0, "0");

        // 验证块定义和引用都存在
        assert_eq!(writer.block_definitions().count(), 1, "应该有 1 个块定义");
        let insert_count = writer
            .drawing()
            .entities()
            .filter(|e| matches!(e.specific, EntityType::Insert(_)))
            .count();
        assert_eq!(insert_count, 1, "应该有 1 个块引用");
    }
}
