//! STL 文件解析器（立体光刻格式）
//!
//! 支持二进制和 ASCII 两种 STL 格式。STL 文件包含三角面片集合，
//! 每个面片包含一个法向量和三个顶点。
//!
//! ## 使用场景
//! - 3D 打印/制造工作流
//! - 从 3D CAD 模型提取几何信息
//! - 3D 模型验证
//!
//! ## 使用示例
//!
//! ```rust,no_run
//! use parser::stl_parser::StlParser;
//!
//! let parser = StlParser::new();
//! let entities = parser.parse_file("model.stl")?;
//! println!("解析到 {} 个三角面", entities.len());
//! # Ok::<(), common_types::error::CadError>(())
//! ```

use common_types::{CadError, EntityMetadata, InternalErrorReason, Point3, RawEntity};
use std::path::Path;

/// STL 解析器
pub struct StlParser;

impl StlParser {
    pub fn new() -> Self {
        Self
    }

    /// 解析 STL 文件
    pub fn parse_file(&self, path: impl AsRef<Path>) -> Result<Vec<RawEntity>, CadError> {
        let path = path.as_ref();
        let mut file = std::fs::File::open(path).map_err(|e| {
            CadError::internal(InternalErrorReason::Panic {
                message: format!("读取 STL 文件失败: {}", e),
            })
        })?;

        let mesh = stl_io::read_stl(&mut file).map_err(|e| {
            CadError::internal(InternalErrorReason::Panic {
                message: format!("解析 STL 失败: {}", e),
            })
        })?;

        let mut entities = Vec::with_capacity(mesh.faces.len());

        for face in &mesh.faces {
            let vertices: [Point3; 3] = [
                [
                    mesh.vertices[face.vertices[0]][0] as f64,
                    mesh.vertices[face.vertices[0]][1] as f64,
                    mesh.vertices[face.vertices[0]][2] as f64,
                ],
                [
                    mesh.vertices[face.vertices[1]][0] as f64,
                    mesh.vertices[face.vertices[1]][1] as f64,
                    mesh.vertices[face.vertices[1]][2] as f64,
                ],
                [
                    mesh.vertices[face.vertices[2]][0] as f64,
                    mesh.vertices[face.vertices[2]][1] as f64,
                    mesh.vertices[face.vertices[2]][2] as f64,
                ],
            ];
            let normal: Point3 = [
                face.normal[0] as f64,
                face.normal[1] as f64,
                face.normal[2] as f64,
            ];

            entities.push(RawEntity::Triangle {
                vertices,
                normal,
                metadata: EntityMetadata::default(),
                semantic: None,
            });
        }

        Ok(entities)
    }

    /// 解析 STL 字节
    pub fn parse_bytes(&self, bytes: &[u8]) -> Result<Vec<RawEntity>, CadError> {
        // stl_io 需要 Seek 且 ASCII probe 会消费 reader，使用 Vec<u8> 确保正确工作
        let mut cursor = std::io::Cursor::new(bytes.to_vec());
        let mesh = stl_io::read_stl(&mut cursor).map_err(|e| {
            CadError::internal(InternalErrorReason::Panic {
                message: format!("解析 STL 字节失败: {}", e),
            })
        })?;

        let mut entities = Vec::with_capacity(mesh.faces.len());

        for face in &mesh.faces {
            let vertices: [Point3; 3] = [
                [
                    mesh.vertices[face.vertices[0]][0] as f64,
                    mesh.vertices[face.vertices[0]][1] as f64,
                    mesh.vertices[face.vertices[0]][2] as f64,
                ],
                [
                    mesh.vertices[face.vertices[1]][0] as f64,
                    mesh.vertices[face.vertices[1]][1] as f64,
                    mesh.vertices[face.vertices[1]][2] as f64,
                ],
                [
                    mesh.vertices[face.vertices[2]][0] as f64,
                    mesh.vertices[face.vertices[2]][1] as f64,
                    mesh.vertices[face.vertices[2]][2] as f64,
                ],
            ];
            let normal: Point3 = [
                face.normal[0] as f64,
                face.normal[1] as f64,
                face.normal[2] as f64,
            ];

            entities.push(RawEntity::Triangle {
                vertices,
                normal,
                metadata: EntityMetadata::default(),
                semantic: None,
            });
        }

        Ok(entities)
    }
}

impl Default for StlParser {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stl_parser_creation() {
        let _parser = StlParser::new();
    }

    #[test]
    fn test_parse_ascii_stl() {
        let stl = r#"solid test
  facet normal 0 0 1
    outer loop
      vertex 0 0 0
      vertex 1 0 0
      vertex 0 1 0
    endloop
  endfacet
  facet normal 0 0 -1
    outer loop
      vertex 0 0 0
      vertex 0 1 0
      vertex 1 0 0
    endloop
  endfacet
endsolid test"#;
        let parser = StlParser::new();
        let entities = parser.parse_bytes(stl.as_bytes()).unwrap();
        assert_eq!(entities.len(), 2, "应该解析到 2 个三角面");

        if let RawEntity::Triangle {
            vertices, normal, ..
        } = &entities[0]
        {
            assert!((normal[2] - 1.0).abs() < 1e-6);
            assert_eq!(vertices.len(), 3);
        } else {
            panic!("Expected Triangle entity");
        }
    }

    #[test]
    fn test_parse_binary_stl() {
        // 创建一个最小的二进制 STL 文件
        let mut data = Vec::new();
        // 80 字节头部（以非 "solid" 开头，确保被识别为二进制格式）
        let mut header = [0u8; 80];
        header[..5].copy_from_slice(b"BSTL "); // Binary STL marker
        data.extend_from_slice(&header);
        // 面片数量
        data.extend_from_slice(&2u32.to_le_bytes());
        // 面片 1: normal(0,0,1)
        data.extend_from_slice(&0.0f32.to_le_bytes());
        data.extend_from_slice(&0.0f32.to_le_bytes());
        data.extend_from_slice(&1.0f32.to_le_bytes());
        // 顶点 1
        data.extend_from_slice(&0.0f32.to_le_bytes());
        data.extend_from_slice(&0.0f32.to_le_bytes());
        data.extend_from_slice(&0.0f32.to_le_bytes());
        // 顶点 2
        data.extend_from_slice(&1.0f32.to_le_bytes());
        data.extend_from_slice(&0.0f32.to_le_bytes());
        data.extend_from_slice(&0.0f32.to_le_bytes());
        // 顶点 3
        data.extend_from_slice(&0.0f32.to_le_bytes());
        data.extend_from_slice(&1.0f32.to_le_bytes());
        data.extend_from_slice(&0.0f32.to_le_bytes());
        // 属性字节数 (0)
        data.extend_from_slice(&0u16.to_le_bytes());
        // 面片 2: normal(0,0,-1)
        data.extend_from_slice(&0.0f32.to_le_bytes());
        data.extend_from_slice(&0.0f32.to_le_bytes());
        data.extend_from_slice(&(-1.0f32).to_le_bytes());
        data.extend_from_slice(&0.0f32.to_le_bytes());
        data.extend_from_slice(&0.0f32.to_le_bytes());
        data.extend_from_slice(&0.0f32.to_le_bytes());
        data.extend_from_slice(&1.0f32.to_le_bytes());
        data.extend_from_slice(&0.0f32.to_le_bytes());
        data.extend_from_slice(&0.0f32.to_le_bytes());
        data.extend_from_slice(&0.0f32.to_le_bytes());
        data.extend_from_slice(&0.0f32.to_le_bytes());
        data.extend_from_slice(&0.0f32.to_le_bytes());
        data.extend_from_slice(&0u16.to_le_bytes());

        let parser = StlParser::new();
        let entities = parser.parse_bytes(&data).unwrap();
        assert_eq!(entities.len(), 2, "应该解析到 2 个三角面");
    }

    #[test]
    fn test_parse_invalid_stl() {
        let parser = StlParser::new();
        let result = parser.parse_bytes(b"not an stl file");
        assert!(result.is_err());
    }
}
