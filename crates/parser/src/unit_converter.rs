//! 单位转换器
//!
//! ## 设计哲学
//!
//! DXF 文件中的 `$INSUNITS` 变量可能设置错误，导致单位解析与实际坐标不符。
//! 本模块实现**智能单位检测与自动校正**：
//!
//! 1. 从 `$INSUNITS` 解析声明单位
//! 2. 基于坐标范围推断实际单位
//! 3. 检测单位不匹配并自动校正
//!
//! ## 使用示例
//!
//! ```text
//! use parser::unit_converter::UnitConverter;
//!
//! let converter = UnitConverter::new(declared_unit, &entities);
//!
//! // 检测单位问题
//! if let Some(warning) = converter.get_warning() {
//!     tracing::warn!("{}", warning);
//! }
//!
//! // 转换坐标到毫米
//! let converted: Vec<RawEntity> = entities
//!     .iter()
//! .map(|e| converter.convert_entity(e))
//!     .collect();
//! ```

use common_types::{LengthUnit, Point2, RawEntity};
use serde::{Deserialize, Serialize};

/// 坐标范围 bounding box
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct BoundingBox {
    pub min_x: f64,
    pub max_x: f64,
    pub min_y: f64,
    pub max_y: f64,
}

impl BoundingBox {
    /// 计算对角线长度
    pub fn diagonal(&self) -> f64 {
        let width = self.max_x - self.min_x;
        let height = self.max_y - self.min_y;
        (width * width + height * height).sqrt()
    }

    /// 获取最大坐标值
    pub fn max_coord(&self) -> f64 {
        self.max_x.abs().max(self.max_y.abs())
    }
}

/// 单位转换结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnitConversionResult {
    /// 声明单位（从 DXF $INSUNITS 解析）
    pub declared_unit: LengthUnit,
    /// 推断单位（基于坐标范围）
    pub inferred_unit: LengthUnit,
    /// 是否检测到单位不匹配
    pub unit_mismatch: bool,
    /// 缩放比例（从推断单位到毫米）
    pub scale_to_mm: f64,
}

/// 单位转换器
///
/// ## 核心功能
///
/// 1. **单位检测**：基于坐标范围推断实际单位
/// 2. **自动校正**：检测单位不匹配时自动使用推断单位
/// 3. **坐标转换**：将所有坐标转换到毫米单位
///
/// ## 启发式规则
///
/// ### 规则 1：米单位检测
/// 如果声明单位是米，但坐标值 > 10000（10 米），实际可能是毫米
/// - 典型场景：建筑图纸，设计师用毫米画，但 $INSUNITS 设为米
///
/// ### 规则 2：毫米单位检测
/// 如果声明单位是毫米，但坐标值 < 10（1 厘米），实际可能是米
/// - 典型场景：地图数据，用米画，但 $INSUNITS 设为毫米
///
/// ### 规则 3：英寸单位检测
/// 如果声明单位是英寸，但坐标值 > 10000（254 米），实际可能是毫米
/// - 典型场景：进口设备图纸，单位设置混乱
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnitConverter {
    /// 声明单位（从 DXF $INSUNITS 解析）
    pub declared_unit: LengthUnit,
    /// 推断单位（基于坐标范围）
    pub inferred_unit: LengthUnit,
    /// 是否检测到单位不匹配
    pub unit_mismatch: bool,
    /// 缩放比例（从推断单位到毫米）
    pub scale_to_mm: f64,
    /// 坐标范围
    pub bounds: BoundingBox,
}

impl UnitConverter {
    /// 创建单位转换器（自动检测单位）
    ///
    /// ## 参数
    /// - `declared_unit`: 从 DXF $INSUNITS 解析的声明单位
    /// - `entities`: 所有实体（用于计算坐标范围）
    ///
    /// ## 示例
    /// ```rust,ignore
    /// let converter = UnitConverter::new(LengthUnit::M, &entities);
    /// ```
    pub fn new(declared_unit: LengthUnit, entities: &[RawEntity]) -> Self {
        let bounds = Self::compute_bounds(entities);
        let inferred_unit = Self::infer_unit_from_bounds(bounds, declared_unit);
        let unit_mismatch = declared_unit != inferred_unit;
        let scale_to_mm = Self::unit_to_mm(inferred_unit);

        Self {
            declared_unit,
            inferred_unit,
            unit_mismatch,
            scale_to_mm,
            bounds,
        }
    }

    /// 计算坐标范围
    fn compute_bounds(entities: &[RawEntity]) -> BoundingBox {
        let mut bounds = BoundingBox::default();
        let mut has_data = false;

        for entity in entities {
            has_data = true;
            match entity {
                RawEntity::Line { start, end, .. } => {
                    bounds.min_x = bounds.min_x.min(start[0]).min(end[0]);
                    bounds.max_x = bounds.max_x.max(start[0]).max(end[0]);
                    bounds.min_y = bounds.min_y.min(start[1]).min(end[1]);
                    bounds.max_y = bounds.max_y.max(start[1]).max(end[1]);
                }
                RawEntity::Polyline { points, .. } => {
                    for pt in points {
                        bounds.min_x = bounds.min_x.min(pt[0]);
                        bounds.max_x = bounds.max_x.max(pt[0]);
                        bounds.min_y = bounds.min_y.min(pt[1]);
                        bounds.max_y = bounds.max_y.max(pt[1]);
                    }
                }
                RawEntity::Arc { center, radius, .. }
                | RawEntity::Circle { center, radius, .. } => {
                    bounds.min_x = bounds.min_x.min(center[0] - radius);
                    bounds.max_x = bounds.max_x.max(center[0] + radius);
                    bounds.min_y = bounds.min_y.min(center[1] - radius);
                    bounds.max_y = bounds.max_y.max(center[1] + radius);
                }
                RawEntity::Text { position, .. } => {
                    bounds.min_x = bounds.min_x.min(position[0]);
                    bounds.max_x = bounds.max_x.max(position[0]);
                    bounds.min_y = bounds.min_y.min(position[1]);
                    bounds.max_y = bounds.max_y.max(position[1]);
                }
                _ => {}
            }
        }

        if !has_data {
            // 默认范围：1 米 x 1 米
            BoundingBox {
                min_x: 0.0,
                max_x: 1000.0,
                min_y: 0.0,
                max_y: 1000.0,
            }
        } else {
            bounds
        }
    }

    /// 基于坐标范围推断实际单位
    fn infer_unit_from_bounds(bounds: BoundingBox, declared: LengthUnit) -> LengthUnit {
        let max_coord = bounds.max_coord();
        let diagonal = bounds.diagonal();

        // 启发式规则
        match declared {
            // 规则 1：声明是米，但坐标>10000，实际可能是毫米
            LengthUnit::M if max_coord > 10000.0 => LengthUnit::Mm,

            // 规则 2：声明是毫米，但坐标<10，实际可能是米
            LengthUnit::Mm if max_coord < 10.0 && diagonal < 100.0 => LengthUnit::M,

            // 规则 3：声明是英寸，但坐标>10000，实际可能是毫米
            LengthUnit::Inch if max_coord > 10000.0 => LengthUnit::Mm,

            // 规则 4：声明是英尺，但坐标>1000，实际可能是毫米
            LengthUnit::Foot if max_coord > 1000.0 => LengthUnit::Mm,

            // 规则 5：声明是厘米，但坐标>10000，实际可能是毫米
            LengthUnit::Cm if max_coord > 10000.0 => LengthUnit::Mm,

            // 规则 6：声明是码，但坐标>5000，实际可能是毫米
            LengthUnit::Yard if max_coord > 5000.0 => LengthUnit::Mm,

            // 规则 7：声明是英里，但坐标>100，实际可能是千米或米
            LengthUnit::Mile if max_coord > 100.0 => LengthUnit::Kilometer,

            // 规则 8：声明是微米，但坐标>1000000，实际可能是毫米
            LengthUnit::Micron if max_coord > 1_000_000.0 => LengthUnit::Mm,

            // 规则 9：声明是千米，但坐标<1，实际可能是米
            LengthUnit::Kilometer if max_coord < 1.0 => LengthUnit::M,

            // 规则 10：声明是点/派卡，但坐标>5000，实际可能是毫米
            LengthUnit::Point | LengthUnit::Pica if max_coord > 5000.0 => LengthUnit::Mm,

            // 默认：使用声明单位
            _ => declared,
        }
    }

    /// 获取单位到毫米的转换比例
    fn unit_to_mm(unit: LengthUnit) -> f64 {
        match unit {
            LengthUnit::M => 1000.0,
            LengthUnit::Cm => 10.0,
            LengthUnit::Mm => 1.0,
            LengthUnit::Inch => 25.4,
            LengthUnit::Foot => 304.8,
            LengthUnit::Yard => 914.4,
            LengthUnit::Mile => 1_609_344.0,
            LengthUnit::Micron => 0.001,
            LengthUnit::Kilometer => 1_000_000.0,
            LengthUnit::Point => 0.352778,
            LengthUnit::Pica => 4.23333,
            LengthUnit::Unspecified => 1.0,
        }
    }

    /// 获取单位警告信息
    ///
    /// ## 返回
    /// - `Some(String)`: 检测到单位不匹配，返回警告信息
    /// - `None`: 单位正常
    pub fn get_warning(&self) -> Option<String> {
        if self.unit_mismatch {
            Some(format!(
                "⚠️ 检测到单位不匹配：\n\
                 \t图纸声明单位：{:?}\n\
                 \t推断实际单位：{:?}\n\
                 \t坐标范围：[{:.2}, {:.2}] × [{:.2}, {:.2}]\n\
                 \t已自动校正，请检查几何尺寸是否正确。",
                self.declared_unit,
                self.inferred_unit,
                self.bounds.min_x,
                self.bounds.max_x,
                self.bounds.min_y,
                self.bounds.max_y,
            ))
        } else {
            None
        }
    }

    /// 转换单个点到毫米
    pub fn convert_point(&self, point: Point2) -> Point2 {
        [point[0] * self.scale_to_mm, point[1] * self.scale_to_mm]
    }

    /// 转换实体到毫米
    pub fn convert_entity(&self, entity: &RawEntity) -> RawEntity {
        match entity {
            RawEntity::Line {
                start,
                end,
                metadata,
                semantic,
            } => RawEntity::Line {
                start: self.convert_point(*start),
                end: self.convert_point(*end),
                metadata: metadata.clone(),
                semantic: semantic.clone(),
            },
            RawEntity::Polyline {
                points,
                closed,
                metadata,
                semantic,
            } => RawEntity::Polyline {
                points: points.iter().map(|p| self.convert_point(*p)).collect(),
                closed: *closed,
                metadata: metadata.clone(),
                semantic: semantic.clone(),
            },
            RawEntity::Arc {
                center,
                radius,
                start_angle,
                end_angle,
                metadata,
                semantic,
            } => RawEntity::Arc {
                center: self.convert_point(*center),
                radius: *radius * self.scale_to_mm,
                start_angle: *start_angle,
                end_angle: *end_angle,
                metadata: metadata.clone(),
                semantic: semantic.clone(),
            },
            RawEntity::Circle {
                center,
                radius,
                metadata,
                semantic,
            } => RawEntity::Circle {
                center: self.convert_point(*center),
                radius: *radius * self.scale_to_mm,
                metadata: metadata.clone(),
                semantic: semantic.clone(),
            },
            RawEntity::Text {
                position,
                content,
                height,
                metadata,
                semantic,
                ..
            } => RawEntity::Text {
                position: self.convert_point(*position),
                content: content.clone(),
                height: *height * self.scale_to_mm,
                rotation: 0.0,
                style_name: None,
                align_left: None,
                align_right: None,
                metadata: metadata.clone(),
                semantic: semantic.clone(),
            },
            RawEntity::Path {
                commands,
                metadata,
                semantic,
            } => {
                // Path 命令中的坐标也需要转换
                let converted_commands = commands
                    .iter()
                    .map(|cmd| match cmd {
                        common_types::PathCommand::MoveTo { x, y } => {
                            let pt = self.convert_point([*x, *y]);
                            common_types::PathCommand::MoveTo { x: pt[0], y: pt[1] }
                        }
                        common_types::PathCommand::LineTo { x, y } => {
                            let pt = self.convert_point([*x, *y]);
                            common_types::PathCommand::LineTo { x: pt[0], y: pt[1] }
                        }
                        common_types::PathCommand::ArcTo {
                            rx,
                            ry,
                            x_axis_rotation,
                            large_arc,
                            sweep,
                            x,
                            y,
                        } => {
                            let pt = self.convert_point([*x, *y]);
                            common_types::PathCommand::ArcTo {
                                rx: *rx * self.scale_to_mm,
                                ry: *ry * self.scale_to_mm,
                                x_axis_rotation: *x_axis_rotation,
                                large_arc: *large_arc,
                                sweep: *sweep,
                                x: pt[0],
                                y: pt[1],
                            }
                        }
                        common_types::PathCommand::Close => common_types::PathCommand::Close,
                    })
                    .collect();

                RawEntity::Path {
                    commands: converted_commands,
                    metadata: metadata.clone(),
                    semantic: semantic.clone(),
                }
            }
            RawEntity::BlockReference {
                block_name,
                insertion_point,
                scale,
                rotation,
                metadata,
                semantic,
            } => {
                RawEntity::BlockReference {
                    block_name: block_name.clone(),
                    insertion_point: self.convert_point(*insertion_point),
                    scale: *scale, // 缩放比例不变
                    rotation: *rotation,
                    metadata: metadata.clone(),
                    semantic: semantic.clone(),
                }
            }
            RawEntity::Dimension {
                dimension_type,
                measurement,
                text,
                definition_points,
                metadata,
                semantic,
            } => RawEntity::Dimension {
                dimension_type: dimension_type.clone(),
                measurement: *measurement * self.scale_to_mm,
                text: text.clone(),
                definition_points: definition_points
                    .iter()
                    .map(|p| self.convert_point(*p))
                    .collect(),
                metadata: metadata.clone(),
                semantic: semantic.clone(),
            },
            RawEntity::Hatch {
                boundary_paths,
                pattern,
                solid_fill,
                metadata,
                semantic,
                scale,
                angle,
            } => {
                // 转换 HATCH 边界路径中的坐标
                let converted_boundaries: Vec<common_types::HatchBoundaryPath> = boundary_paths
                    .iter()
                    .map(|boundary| match boundary {
                        common_types::HatchBoundaryPath::Polyline {
                            points,
                            closed,
                            bulges,
                        } => common_types::HatchBoundaryPath::Polyline {
                            points: points.iter().map(|p| self.convert_point(*p)).collect(),
                            closed: *closed,
                            bulges: bulges.clone(),
                        },
                        common_types::HatchBoundaryPath::Arc {
                            center,
                            radius,
                            start_angle,
                            end_angle,
                            ccw,
                        } => common_types::HatchBoundaryPath::Arc {
                            center: self.convert_point(*center),
                            radius: *radius * self.scale_to_mm,
                            start_angle: *start_angle,
                            end_angle: *end_angle,
                            ccw: *ccw,
                        },
                        common_types::HatchBoundaryPath::EllipseArc {
                            center,
                            major_axis,
                            minor_axis_ratio,
                            start_angle,
                            end_angle,
                            ccw,
                            extrusion_direction,
                        } => common_types::HatchBoundaryPath::EllipseArc {
                            center: self.convert_point(*center),
                            major_axis: self.convert_point(*major_axis),
                            minor_axis_ratio: *minor_axis_ratio,
                            start_angle: *start_angle,
                            end_angle: *end_angle,
                            ccw: *ccw,
                            extrusion_direction: *extrusion_direction,
                        },
                        common_types::HatchBoundaryPath::Spline {
                            control_points,
                            knots,
                            degree,
                            weights,
                            fit_points,
                            flags,
                        } => common_types::HatchBoundaryPath::Spline {
                            control_points: control_points
                                .iter()
                                .map(|p| self.convert_point(*p))
                                .collect(),
                            knots: knots.clone(),
                            degree: *degree,
                            weights: weights.clone(),
                            fit_points: fit_points
                                .as_ref()
                                .map(|fp| fp.iter().map(|p| self.convert_point(*p)).collect()),
                            flags: *flags,
                        },
                    })
                    .collect();

                RawEntity::Hatch {
                    boundary_paths: converted_boundaries,
                    pattern: pattern.clone(),
                    solid_fill: *solid_fill,
                    metadata: metadata.clone(),
                    semantic: semantic.clone(),
                    scale: *scale, // P0-NEW-14 修复：传递 scale
                    angle: *angle, // P0-NEW-14 修复：传递 angle
                }
            }
            RawEntity::XRef { .. } => {
                // P1-1: XREF 外部参照支持 - 待完整实现
                // 目前仅做类型标记，不进行几何处理
                todo!("XREF 外部参照单位转换 - 需要后续实现外部文件加载和单位转换")
            }
            RawEntity::Point {
                position,
                metadata,
                semantic,
            } => RawEntity::Point {
                position: self.convert_point(*position),
                metadata: metadata.clone(),
                semantic: semantic.clone(),
            },
            RawEntity::Image {
                image_def,
                position,
                size,
                metadata,
                semantic,
            } => RawEntity::Image {
                image_def: image_def.clone(),
                position: self.convert_point(*position),
                size: [size[0] * self.scale_to_mm, size[1] * self.scale_to_mm],
                metadata: metadata.clone(),
                semantic: semantic.clone(),
            },
            RawEntity::Attribute {
                tag,
                value,
                position,
                height,
                rotation,
                metadata,
                semantic,
            } => RawEntity::Attribute {
                tag: tag.clone(),
                value: value.clone(),
                position: self.convert_point(*position),
                height: *height * self.scale_to_mm,
                rotation: *rotation,
                metadata: metadata.clone(),
                semantic: semantic.clone(),
            },
            RawEntity::AttributeDefinition {
                tag,
                default_value,
                prompt,
                position,
                height,
                rotation,
                metadata,
                semantic,
            } => RawEntity::AttributeDefinition {
                tag: tag.clone(),
                default_value: default_value.clone(),
                prompt: prompt.clone(),
                position: self.convert_point(*position),
                height: *height * self.scale_to_mm,
                rotation: *rotation,
                metadata: metadata.clone(),
                semantic: semantic.clone(),
            },
            RawEntity::Leader {
                points,
                annotation_text,
                metadata,
                semantic,
            } => RawEntity::Leader {
                points: points.iter().map(|p| self.convert_point(*p)).collect(),
                annotation_text: annotation_text.clone(),
                metadata: metadata.clone(),
                semantic: semantic.clone(),
            },
            RawEntity::Ray {
                start,
                direction,
                metadata,
                semantic,
            } => RawEntity::Ray {
                start: self.convert_point(*start),
                direction: [
                    direction[0] * self.scale_to_mm,
                    direction[1] * self.scale_to_mm,
                ],
                metadata: metadata.clone(),
                semantic: semantic.clone(),
            },
            RawEntity::MLine {
                center_line,
                closed,
                style_name,
                scale_factor,
                metadata,
                semantic,
            } => RawEntity::MLine {
                center_line: center_line.iter().map(|p| self.convert_point(*p)).collect(),
                closed: *closed,
                style_name: style_name.clone(),
                scale_factor: *scale_factor,
                metadata: metadata.clone(),
                semantic: semantic.clone(),
            },
            // Triangle 使用 3D 单位转换
            RawEntity::Triangle {
                vertices,
                normal,
                metadata,
                semantic,
            } => RawEntity::Triangle {
                vertices: [
                    [
                        vertices[0][0] * self.scale_to_mm,
                        vertices[0][1] * self.scale_to_mm,
                        vertices[0][2] * self.scale_to_mm,
                    ],
                    [
                        vertices[1][0] * self.scale_to_mm,
                        vertices[1][1] * self.scale_to_mm,
                        vertices[1][2] * self.scale_to_mm,
                    ],
                    [
                        vertices[2][0] * self.scale_to_mm,
                        vertices[2][1] * self.scale_to_mm,
                        vertices[2][2] * self.scale_to_mm,
                    ],
                ],
                normal: *normal, // 法向量是方向，不需要缩放
                metadata: metadata.clone(),
                semantic: semantic.clone(),
            },
        }
    }

    /// 转换实体列表到毫米
    pub fn convert_entities(&self, entities: &[RawEntity]) -> Vec<RawEntity> {
        entities.iter().map(|e| self.convert_entity(e)).collect()
    }

    /// 获取转换结果摘要
    pub fn summary(&self) -> String {
        format!(
            "UnitConverter {{\n\
             \t声明单位：{:?}\n\
             \t推断单位：{:?}\n\
             \t单位不匹配：{}\n\
             \t缩放比例：{:.4}\n\
             \t坐标范围：[{:.2}, {:.2}] × [{:.2}, {:.2}]\n\
             }}",
            self.declared_unit,
            self.inferred_unit,
            self.unit_mismatch,
            self.scale_to_mm,
            self.bounds.min_x,
            self.bounds.max_x,
            self.bounds.min_y,
            self.bounds.max_y,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unit_inference_m_to_mm() {
        // 声明单位是米，但坐标值很大（实际是毫米）
        // 10000.0 米 = 10 公里，这在实际建筑图纸中不合理，应推断为毫米（10 米）
        let entities = vec![RawEntity::Line {
            start: [0.0, 0.0],
            end: [10001.0, 0.0], // > 10000 触发推断规则
            metadata: Default::default(),
            semantic: None,
        }];

        let converter = UnitConverter::new(LengthUnit::M, &entities);
        assert_eq!(converter.inferred_unit, LengthUnit::Mm);
        assert!(converter.unit_mismatch);
    }

    #[test]
    fn test_unit_inference_mm_to_m() {
        // 声明单位是毫米，但坐标值很小（实际是米）
        let entities = vec![RawEntity::Line {
            start: [0.0, 0.0],
            end: [5.0, 0.0],
            metadata: Default::default(),
            semantic: None,
        }];

        let converter = UnitConverter::new(LengthUnit::Mm, &entities);
        assert_eq!(converter.inferred_unit, LengthUnit::M);
        assert!(converter.unit_mismatch);
    }

    #[test]
    fn test_unit_conversion() {
        let entities = vec![RawEntity::Line {
            start: [0.0, 0.0],
            end: [10.0, 0.0],
            metadata: Default::default(),
            semantic: None,
        }];

        let converter = UnitConverter::new(LengthUnit::M, &entities);
        let converted = converter.convert_entities(&entities);

        if let RawEntity::Line { start, end, .. } = &converted[0] {
            assert!((start[0] - 0.0).abs() < 1e-10);
            assert!((end[0] - 10000.0).abs() < 1e-10); // 10 米 = 10000 毫米
        } else {
            panic!("Expected Line entity");
        }
    }
}
