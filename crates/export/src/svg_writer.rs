//! SVG 导出器
//!
//! 将 `Vec<RawEntity>` 导出为 SVG XML 格式。
//!
//! ## 映射规则
//!
//! | RawEntity | SVG 元素 |
//! |-----------|----------|
//! | `Line` | `<line>` |
//! | `Polyline` | `<polyline>` / `<polygon>` |
//! | `Circle` | `<circle>` |
//! | `Arc` | `<path d="M... A...">` |
//! | `Text` | `<text>` |
//! | `Path` | `<path>` |
//! | `Triangle` | `<polygon>` (投影到 XY 平面) |
//!
//! ## 使用示例
//!
//! ```rust
//! use export::svg_writer::SvgWriter;
//!
//! let writer = SvgWriter::new();
//! let svg = writer.write(&[]);
//! assert!(svg.starts_with("<?xml"));
//! ```

use common_types::{PathCommand, RawEntity};

/// SVG 导出配置
#[derive(Debug, Clone)]
pub struct SvgConfig {
    /// 画布宽度
    pub width: Option<f64>,
    /// 画布高度
    pub height: Option<f64>,
    /// viewBox
    pub view_box: Option<(f64, f64, f64, f64)>,
    /// 图层可见性过滤
    pub layer_filter: Option<Vec<String>>,
    /// 是否美化输出（缩进+换行）
    pub pretty: bool,
}

impl Default for SvgConfig {
    fn default() -> Self {
        Self {
            width: None,
            height: None,
            view_box: None,
            layer_filter: None,
            pretty: true,
        }
    }
}

/// SVG 导出器
pub struct SvgWriter {
    config: SvgConfig,
}

impl SvgWriter {
    pub fn new() -> Self {
        Self {
            config: SvgConfig::default(),
        }
    }

    pub fn with_config(mut self, config: SvgConfig) -> Self {
        self.config = config;
        self
    }

    /// 将实体列表导出为 SVG 字符串
    pub fn write(&self, entities: &[RawEntity]) -> String {
        let mut svg = String::new();
        svg.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");

        // 计算边界框（如果未指定 viewBox）
        let view_box = self
            .config
            .view_box
            .unwrap_or_else(|| Self::compute_view_box(entities));
        let width = self.config.width.unwrap_or(view_box.2 - view_box.0);
        let height = self.config.height.unwrap_or(view_box.3 - view_box.1);

        svg.push_str(&format!(
            "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{}\" height=\"{}\" viewBox=\"{} {} {} {}\">\n",
            width, height, view_box.0, view_box.1, view_box.2, view_box.3,
        ));

        let indent = if self.config.pretty { "  " } else { "" };

        for entity in entities {
            if let Some(ref layers) = self.config.layer_filter {
                if let Some(layer) = entity.layer() {
                    if !layers.contains(&layer.to_string()) {
                        continue;
                    }
                }
            }
            if let Some(elem) = Self::entity_to_svg(entity) {
                svg.push_str(indent);
                svg.push_str(&elem);
                svg.push('\n');
            }
        }

        svg.push_str("</svg>\n");
        svg
    }

    /// 计算所有实体的边界框
    fn compute_view_box(entities: &[RawEntity]) -> (f64, f64, f64, f64) {
        let mut min_x = f64::MAX;
        let mut min_y = f64::MAX;
        let mut max_x = f64::MIN;
        let mut max_y = f64::MIN;

        for entity in entities {
            match entity {
                RawEntity::Line { start, end, .. } => {
                    min_x = min_x.min(start[0]).min(end[0]);
                    min_y = min_y.min(start[1]).min(end[1]);
                    max_x = max_x.max(start[0]).max(end[0]);
                    max_y = max_y.max(start[1]).max(end[1]);
                }
                RawEntity::Polyline { points, .. } => {
                    for p in points {
                        min_x = min_x.min(p[0]);
                        min_y = min_y.min(p[1]);
                        max_x = max_x.max(p[0]);
                        max_y = max_y.max(p[1]);
                    }
                }
                RawEntity::Arc { center, radius, .. }
                | RawEntity::Circle { center, radius, .. } => {
                    min_x = min_x.min(center[0] - radius);
                    min_y = min_y.min(center[1] - radius);
                    max_x = max_x.max(center[0] + radius);
                    max_y = max_y.max(center[1] + radius);
                }
                RawEntity::Text { position, .. } => {
                    min_x = min_x.min(position[0]);
                    min_y = min_y.min(position[1]);
                    max_x = max_x.max(position[0]);
                    max_y = max_y.max(position[1]);
                }
                RawEntity::Path { commands, .. } => {
                    for cmd in commands {
                        match cmd {
                            PathCommand::MoveTo { x, y }
                            | PathCommand::LineTo { x, y }
                            | PathCommand::ArcTo { x, y, .. } => {
                                min_x = min_x.min(*x);
                                min_y = min_y.min(*y);
                                max_x = max_x.max(*x);
                                max_y = max_y.max(*y);
                            }
                            PathCommand::Close => {}
                        }
                    }
                }
                RawEntity::Triangle { vertices, .. } => {
                    for v in vertices {
                        min_x = min_x.min(v[0]);
                        min_y = min_y.min(v[1]);
                        max_x = max_x.max(v[0]);
                        max_y = max_y.max(v[1]);
                    }
                }
                _ => {}
            }
        }

        // 添加 10% 的边距
        let padding_x = (max_x - min_x) * 0.1;
        let padding_y = (max_y - min_y) * 0.1;

        if min_x.is_finite() {
            (
                min_x - padding_x,
                min_y - padding_y,
                max_x + padding_x,
                max_y + padding_y,
            )
        } else {
            (0.0, 0.0, 100.0, 100.0)
        }
    }

    /// 将单个实体转换为 SVG 元素字符串
    fn entity_to_svg(entity: &RawEntity) -> Option<String> {
        let layer_attr = entity
            .layer()
            .map(|l| format!(" data-layer=\"{}\"", l))
            .unwrap_or_default();

        match entity {
            RawEntity::Line { start, end, .. } => Some(format!(
                "<line x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\" stroke=\"black\" stroke-width=\"0.5\"{}/>",
                start[0], start[1], end[0], end[1], layer_attr,
            )),
            RawEntity::Polyline {
                points, closed, ..
            } => {
                if points.is_empty() {
                    return None;
                }
                let pts: Vec<String> = points
                    .iter()
                    .map(|p| format!("{},{}", p[0], p[1]))
                    .collect();
                let tag = if *closed { "polygon" } else { "polyline" };
                Some(format!(
                    "<{} points=\"{}\" fill=\"none\" stroke=\"black\" stroke-width=\"0.5\"{}/>",
                    tag,
                    pts.join(" "),
                    layer_attr,
                ))
            }
            RawEntity::Circle { center, radius, .. } => Some(format!(
                "<circle cx=\"{}\" cy=\"{}\" r=\"{}\" fill=\"none\" stroke=\"black\" stroke-width=\"0.5\"{}/>",
                center[0], center[1], radius, layer_attr,
            )),
            RawEntity::Arc {
                center,
                radius,
                start_angle,
                end_angle,
                ..
            } => {
                // 将角度转换为弧度
                let start_rad = start_angle.to_radians();
                let end_rad = end_angle.to_radians();
                let x1 = center[0] + radius * start_rad.cos();
                let y1 = center[1] + radius * start_rad.sin();
                let x2 = center[0] + radius * end_rad.cos();
                let y2 = center[1] + radius * end_rad.sin();
                let sweep = if *end_angle > *start_angle { 1 } else { 0 };
                let large_arc = if (end_rad - start_rad).abs() > std::f64::consts::PI {
                    1
                } else {
                    0
                };
                Some(format!(
                    "<path d=\"M {} {} A {} {} 0 {} {} {} {}\" fill=\"none\" stroke=\"black\" stroke-width=\"0.5\"{}/>",
                    x1, y1, radius, radius, large_arc, sweep, x2, y2, layer_attr,
                ))
            }
            RawEntity::Text {
                position,
                content,
                height,
                ..
            } => Some(format!(
                "<text x=\"{}\" y=\"{}\" font-size=\"{}\" fill=\"black\">{}</text>",
                position[0],
                position[1],
                height,
                escape_xml(content),
            )),
            RawEntity::Path { commands, .. } => {
                if commands.is_empty() {
                    return None;
                }
                let d: Vec<String> = commands
                    .iter()
                    .map(|cmd| match cmd {
                        PathCommand::MoveTo { x, y } => format!("M {} {}", x, y),
                        PathCommand::LineTo { x, y } => format!("L {} {}", x, y),
                        PathCommand::ArcTo {
                            rx,
                            ry,
                            x_axis_rotation,
                            large_arc,
                            sweep,
                            x,
                            y,
                        } => format!(
                            "A {} {} {} {} {} {} {}",
                            rx, ry, x_axis_rotation,
                            if *large_arc { 1 } else { 0 },
                            if *sweep { 1 } else { 0 },
                            x, y,
                        ),
                        PathCommand::Close => "Z".to_string(),
                    })
                    .collect();
                Some(format!(
                    "<path d=\"{}\" fill=\"none\" stroke=\"black\" stroke-width=\"0.5\"{}/>",
                    d.join(" "),
                    layer_attr,
                ))
            }
            RawEntity::Triangle { vertices, .. } => {
                // 投影到 XY 平面
                let pts: Vec<String> = vertices
                    .iter()
                    .map(|v| format!("{},{}", v[0], v[1]))
                    .collect();
                Some(format!(
                    "<polygon points=\"{}\" fill=\"none\" stroke=\"black\" stroke-width=\"0.5\"{}/>",
                    pts.join(" "),
                    layer_attr,
                ))
            }
            // 以下实体类型不适合直接映射到 SVG，跳过
            RawEntity::BlockReference { .. }
            | RawEntity::Dimension { .. }
            | RawEntity::Hatch { .. }
            | RawEntity::XRef { .. }
            | RawEntity::Point { .. }
            | RawEntity::Image { .. }
            | RawEntity::Attribute { .. }
            | RawEntity::AttributeDefinition { .. }
            | RawEntity::Leader { .. }
            | RawEntity::Ray { .. }
            | RawEntity::MLine { .. } => None,
        }
    }
}

impl Default for SvgWriter {
    fn default() -> Self {
        Self::new()
    }
}

/// 转义 XML 特殊字符
fn escape_xml(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

#[cfg(test)]
mod tests {
    use super::*;
    use common_types::EntityMetadata;

    #[test]
    fn test_svg_writer_empty() {
        let writer = SvgWriter::new();
        let svg = writer.write(&[]);
        assert!(svg.contains("<?xml"));
        assert!(svg.contains("<svg"));
        assert!(svg.contains("</svg>"));
    }

    #[test]
    fn test_svg_writer_line() {
        let writer = SvgWriter::new();
        let entities = vec![RawEntity::Line {
            start: [0.0, 0.0],
            end: [100.0, 100.0],
            metadata: EntityMetadata::default(),
            semantic: None,
        }];
        let svg = writer.write(&entities);
        assert!(svg.contains("<line"));
        assert!(svg.contains("x1=\"0\""));
        assert!(svg.contains("x2=\"100\""));
    }

    #[test]
    fn test_svg_writer_circle() {
        let writer = SvgWriter::new();
        let entities = vec![RawEntity::Circle {
            center: [50.0, 50.0],
            radius: 25.0,
            metadata: EntityMetadata::default(),
            semantic: None,
        }];
        let svg = writer.write(&entities);
        assert!(svg.contains("<circle"));
        assert!(svg.contains("cx=\"50\""));
        assert!(svg.contains("r=\"25\""));
    }

    #[test]
    fn test_svg_writer_text() {
        let writer = SvgWriter::new();
        let entities = vec![RawEntity::Text {
            position: [10.0, 20.0],
            content: "Hello <World>".to_string(),
            height: 12.0,
            rotation: 0.0,
            style_name: None,
            align_left: None,
            align_right: None,
            metadata: EntityMetadata::default(),
            semantic: None,
        }];
        let svg = writer.write(&entities);
        assert!(svg.contains("<text"));
        assert!(svg.contains("&lt;World&gt;")); // 应转义
    }

    #[test]
    fn test_svg_writer_triangle() {
        let writer = SvgWriter::new();
        let entities = vec![RawEntity::Triangle {
            vertices: [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [5.0, 10.0, 0.0]],
            normal: [0.0, 0.0, 1.0],
            metadata: EntityMetadata::default(),
            semantic: None,
        }];
        let svg = writer.write(&entities);
        assert!(svg.contains("<polygon"));
    }

    #[test]
    fn test_escape_xml() {
        assert_eq!(escape_xml("<test>"), "&lt;test&gt;");
        assert_eq!(escape_xml("a&b"), "a&amp;b");
    }
}
