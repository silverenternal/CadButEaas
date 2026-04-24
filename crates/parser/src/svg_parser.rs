//! SVG 文件解析器
//!
//! 将 SVG 文件解析为 `Vec<RawEntity>`，用于后续拓扑建模和验证。
//!
//! ## 支持映射
//!
//! | SVG 元素 | RawEntity |
//! |----------|-----------|
//! | `<line>` | `Path` (LineTo) |
//! | `<polyline>` | `Path` |
//! | `<polygon>` | `Path` (closed) |
//! | `<circle>` | `Circle` |
//! | `<rect>` | `Path` (4 点闭合路径) |
//! | `<path>` | `Path` |
//! | `<text>` | `Text` |
//!
//! ## 使用示例
//!
//! ```rust,no_run
//! use parser::svg_parser::SvgParser;
//!
//! let parser = SvgParser::new();
//! let entities = parser.parse_file("drawing.svg")?;
//! # Ok::<(), common_types::error::CadError>(())
//! ```

use common_types::{CadError, EntityMetadata, InternalErrorReason, PathCommand, Point2, RawEntity};
use std::path::Path;
use usvg::tiny_skia_path::PathSegment;

/// SVG 解析器
pub struct SvgParser;

impl SvgParser {
    pub fn new() -> Self {
        Self
    }

    /// 解析 SVG 文件
    pub fn parse_file(&self, path: impl AsRef<Path>) -> Result<Vec<RawEntity>, CadError> {
        let path = path.as_ref();
        let svg_data = std::fs::read(path).map_err(|e| {
            CadError::internal(InternalErrorReason::Panic {
                message: format!("读取 SVG 文件失败: {}", e),
            })
        })?;
        self.parse_bytes(&svg_data)
    }

    /// 解析 SVG 字节
    pub fn parse_bytes(&self, bytes: &[u8]) -> Result<Vec<RawEntity>, CadError> {
        let svg_str = String::from_utf8(bytes.to_vec()).map_err(|e| {
            CadError::internal(InternalErrorReason::Panic {
                message: format!("SVG 文件不是有效的 UTF-8: {}", e),
            })
        })?;

        let tree = usvg::Tree::from_str(&svg_str, &usvg::Options::default()).map_err(|e| {
            CadError::internal(InternalErrorReason::Panic {
                message: format!("解析 SVG 失败: {}", e),
            })
        })?;

        let mut entities = Vec::new();
        Self::walk_group(tree.root(), &mut entities);
        Ok(entities)
    }

    /// 递归遍历 Group 节点
    fn walk_group(group: &usvg::Group, entities: &mut Vec<RawEntity>) {
        for child in group.children() {
            match child {
                usvg::Node::Path(path) => {
                    Self::process_path(path, entities);
                }
                usvg::Node::Image(image) => {
                    Self::process_image(image, entities);
                }
                usvg::Node::Text(text) => {
                    Self::process_text(text, entities);
                }
                usvg::Node::Group(inner) => {
                    Self::walk_group(inner, entities);
                }
            }
        }
    }

    /// 处理 Path 节点
    fn process_path(path: &usvg::Path, entities: &mut Vec<RawEntity>) {
        let data = path.data();
        let mut commands: Vec<PathCommand> = Vec::new();

        for segment in data.segments() {
            match segment {
                PathSegment::MoveTo(pt) => {
                    commands.push(PathCommand::MoveTo {
                        x: pt.x as f64,
                        y: pt.y as f64,
                    });
                }
                PathSegment::LineTo(pt) => {
                    commands.push(PathCommand::LineTo {
                        x: pt.x as f64,
                        y: pt.y as f64,
                    });
                }
                PathSegment::QuadTo(_p0, p1) => {
                    // 二次贝塞尔曲线，简化为直线到终点
                    commands.push(PathCommand::LineTo {
                        x: p1.x as f64,
                        y: p1.y as f64,
                    });
                }
                PathSegment::CubicTo(_p0, _p1, p2) => {
                    // 三次贝塞尔曲线，简化为直线到终点
                    commands.push(PathCommand::LineTo {
                        x: p2.x as f64,
                        y: p2.y as f64,
                    });
                }
                PathSegment::Close => {
                    commands.push(PathCommand::Close);
                }
            }
        }

        if !commands.is_empty() {
            entities.push(RawEntity::Path {
                commands,
                metadata: EntityMetadata::default(),
                semantic: None,
            });
        }
    }

    /// 处理 Image 节点
    fn process_image(_image: &usvg::Image, _entities: &mut Vec<RawEntity>) {
        // usvg 已经将图像解码，但 RawEntity 没有对应的 Image3D 类型
        // 这里暂时跳过，后续可扩展
    }

    /// 处理 Text 节点
    fn process_text(text: &usvg::Text, entities: &mut Vec<RawEntity>) {
        // usvg 的 Text 包含已解析的文字内容
        for chunk in text.chunks() {
            if let Some(span) = chunk.spans().first() {
                let content = chunk.text().to_string();
                let transform = text.abs_transform();
                let position: Point2 = [transform.tx as f64, transform.ty as f64];
                let height = span.font_size().get() as f64;

                entities.push(RawEntity::Text {
                    position,
                    content,
                    height,
                    rotation: 0.0,
                    style_name: None,
                    align_left: None,
                    align_right: None,
                    metadata: EntityMetadata::default(),
                    semantic: None,
                });
            }
        }
    }
}

impl Default for SvgParser {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_svg_parser_creation() {
        let _parser = SvgParser::new();
    }

    #[test]
    fn test_parse_svg_line() {
        let svg = r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
            <line x1="10" y1="10" x2="90" y2="90" />
        </svg>"#;
        let parser = SvgParser::new();
        let entities = parser.parse_bytes(svg.as_bytes()).unwrap();
        let paths: Vec<_> = entities
            .iter()
            .filter(|e| matches!(e, RawEntity::Path { .. }))
            .collect();
        assert!(!paths.is_empty(), "应该解析到路径实体");
    }

    #[test]
    fn test_parse_invalid_svg() {
        let parser = SvgParser::new();
        let result = parser.parse_bytes(b"not an svg file");
        assert!(result.is_err());
    }
}
