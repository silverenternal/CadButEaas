//! 光栅图片加载器
//!
//! 支持从文件或字节流加载 PNG/JPG/BMP/TIFF/WebP 格式，
//! 输出 `image::DynamicImage` 直接对接 `VectorizeService`。

use std::path::Path;

use image::{DynamicImage, GenericImageView};
use thiserror::Error;

use crate::format_detect::{detect_raster_format, RasterFormat};

/// 光栅图片加载错误
#[derive(Error, Debug)]
pub enum RasterError {
    #[error("文件不存在：{0}")]
    FileNotFound(String),

    #[error("文件读取失败：{0}")]
    IoError(#[from] std::io::Error),

    #[error("图片解码失败：{0}")]
    DecodeError(String),

    #[error("不支持的图片格式")]
    UnsupportedFormat,

    #[error("文件过大：{size} 字节超过限制 {limit} 字节")]
    FileTooLarge { size: usize, limit: usize },

    #[error("图片尺寸无效：{width}x{height}")]
    InvalidDimensions { width: u32, height: u32 },
}

/// 光栅图片元数据
#[derive(Debug, Clone)]
pub struct RasterImageInfo {
    /// 图片宽度（像素）
    pub width: u32,
    /// 图片高度（像素）
    pub height: u32,
    /// 水平 DPI（如果从 EXIF 提取到）
    pub dpi_x: Option<f64>,
    /// 垂直 DPI（如果从 EXIF 提取到）
    pub dpi_y: Option<f64>,
    /// 检测到的图片格式
    pub format: RasterFormat,
    /// 源文件路径（如果是从文件加载）
    pub source_path: Option<String>,
}

impl RasterImageInfo {
    pub fn new(width: u32, height: u32, format: RasterFormat) -> Self {
        Self {
            width,
            height,
            dpi_x: None,
            dpi_y: None,
            format,
            source_path: None,
        }
    }
}

/// 光栅图片加载器
pub struct RasterLoader {
    /// 最大允许文件大小（字节），默认 100MB
    pub max_file_size: usize,
    /// 最小允许图片尺寸（像素），默认 1x1
    pub min_dimension: u32,
}

impl RasterLoader {
    /// 创建新的加载器（使用默认限制）
    pub fn new() -> Self {
        Self {
            max_file_size: 100 * 1024 * 1024, // 100MB
            min_dimension: 1,
        }
    }

    /// 从文件路径加载图片
    ///
    /// # 流程
    /// 1. 检查文件存在性和大小
    /// 2. 读取文件字节
    /// 3. 检测格式
    /// 4. 解码为 DynamicImage
    /// 5. 提取元数据（宽高、DPI）
    pub fn from_file(
        path: impl AsRef<Path>,
    ) -> Result<(DynamicImage, RasterImageInfo), RasterError> {
        let path = path.as_ref();

        if !path.exists() {
            return Err(RasterError::FileNotFound(path.display().to_string()));
        }

        let metadata = std::fs::metadata(path)?;
        let file_size = metadata.len() as usize;

        if file_size > 100 * 1024 * 1024 {
            return Err(RasterError::FileTooLarge {
                size: file_size,
                limit: 100 * 1024 * 1024,
            });
        }

        let bytes = std::fs::read(path)?;
        let filename = path.file_name().and_then(|n| n.to_str());

        let (image, info) = Self::from_bytes_internal(&bytes, filename)?;

        let mut info = info;
        info.source_path = Some(path.display().to_string());

        Ok((image, info))
    }

    /// 从字节数据加载图片
    ///
    /// # 参数
    /// * `bytes` - 图片字节数据
    /// * `format_hint` - 可选的文件扩展名提示（如 "png"、".jpg"）
    pub fn from_bytes(
        bytes: &[u8],
        format_hint: Option<&str>,
    ) -> Result<(DynamicImage, RasterImageInfo), RasterError> {
        Self::from_bytes_internal(bytes, format_hint)
    }

    /// 内部加载逻辑（复用 from_file 和 from_bytes）
    fn from_bytes_internal(
        bytes: &[u8],
        format_hint: Option<&str>,
    ) -> Result<(DynamicImage, RasterImageInfo), RasterError> {
        if bytes.is_empty() {
            return Err(RasterError::DecodeError("空文件数据".to_string()));
        }

        // 格式检测
        let format = detect_raster_format(bytes, format_hint).ok_or_else(|| {
            tracing::warn!("无法识别图片格式，格式提示：{:?}", format_hint);
            RasterError::UnsupportedFormat
        })?;

        tracing::debug!(
            "检测到图片格式：{:?}，数据大小：{} 字节",
            format,
            bytes.len()
        );

        // 使用 image::ImageReader 自动解码
        let cursor = std::io::Cursor::new(bytes);
        let image = image::ImageReader::new(cursor)
            .with_guessed_format()
            .map_err(|e| RasterError::DecodeError(format!("无法识别图片格式：{}", e)))?
            .decode()
            .map_err(|e| RasterError::DecodeError(format!("图片解码失败：{}", e)))?;

        let (width, height) = image.dimensions();

        if width == 0 || height == 0 {
            return Err(RasterError::InvalidDimensions { width, height });
        }

        let info = RasterImageInfo::new(width, height, format);

        // DPI 信息需从 EXIF 提取（image crate 0.25 的 EXIF 支持有限）
        // 当前版本不自动提取 DPI，使用者可通过 VectorizeConfig 手动设置

        Ok((image, info))
    }
}

impl Default for RasterLoader {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_loader_new_defaults() {
        let loader = RasterLoader::new();
        assert_eq!(loader.max_file_size, 100 * 1024 * 1024);
        assert_eq!(loader.min_dimension, 1);
    }

    #[test]
    fn test_from_file_not_found() {
        let result = RasterLoader::from_file("/nonexistent/path/image.png");
        assert!(matches!(result, Err(RasterError::FileNotFound(_))));
    }

    #[test]
    fn test_from_bytes_empty() {
        let result = RasterLoader::from_bytes(&[], None);
        assert!(matches!(result, Err(RasterError::DecodeError(_))));
    }

    #[test]
    fn test_from_bytes_invalid_format() {
        let result = RasterLoader::from_bytes(b"this is not an image", None);
        assert!(matches!(result, Err(RasterError::UnsupportedFormat)));
    }

    #[test]
    fn test_raster_image_info() {
        let info = RasterImageInfo::new(800, 600, RasterFormat::Png);
        assert_eq!(info.width, 800);
        assert_eq!(info.height, 600);
        assert_eq!(info.dpi_x, None);
        assert_eq!(info.format, RasterFormat::Png);
    }
}
