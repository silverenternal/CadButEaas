//! 光栅图片加载器
//!
//! 支持从文件或字节流加载 PNG/JPG/BMP/TIFF/WebP 格式，
//! 输出 `image::DynamicImage` 直接对接 `VectorizeService`。

use std::io::Cursor;
use std::path::Path;

use exif::Reader as ExifReader;
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
    /// DPI 来源（png_phys/tiff_resolution/jpeg_jfif/jpeg_exif）
    pub dpi_source: Option<String>,
    /// DPI 是否可用于毫米尺度恢复。
    pub dpi_trusted: bool,
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
            dpi_source: None,
            dpi_trusted: false,
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

        let mut info = RasterImageInfo::new(width, height, format);

        // 从格式元数据提取 DPI 信息
        if let Some((dpi_x, dpi_y, source, trusted)) = extract_dpi(bytes, format) {
            tracing::debug!("从 {} 提取到 DPI: {} x {}", source, dpi_x, dpi_y);
            info.dpi_x = Some(dpi_x);
            info.dpi_y = Some(dpi_y);
            info.dpi_source = Some(source);
            info.dpi_trusted = trusted;
        }

        Ok((image, info))
    }
}

impl Default for RasterLoader {
    fn default() -> Self {
        Self::new()
    }
}

fn extract_dpi(bytes: &[u8], format: RasterFormat) -> Option<(f64, f64, String, bool)> {
    match format {
        RasterFormat::Png => extract_dpi_from_png_phys(bytes)
            .map(|(x, y, trusted)| (x, y, "png_phys".to_string(), trusted)),
        RasterFormat::Tiff => extract_dpi_from_tiff(bytes)
            .map(|(x, y, trusted)| (x, y, "tiff_resolution".to_string(), trusted)),
        RasterFormat::Jpeg => extract_dpi_from_jpeg_jfif(bytes)
            .map(|(x, y, trusted)| (x, y, "jpeg_jfif".to_string(), trusted))
            .or_else(|| {
                extract_dpi_from_exif(bytes).map(|(x, y)| (x, y, "jpeg_exif".to_string(), true))
            }),
        _ => extract_dpi_from_exif(bytes).map(|(x, y)| (x, y, "exif".to_string(), true)),
    }
}

fn extract_dpi_from_png_phys(bytes: &[u8]) -> Option<(f64, f64, bool)> {
    const PNG_SIG: &[u8; 8] = b"\x89PNG\r\n\x1a\n";
    if bytes.len() < 8 || &bytes[..8] != PNG_SIG {
        return None;
    }

    let mut offset = 8usize;
    while offset.checked_add(12)? <= bytes.len() {
        let length = u32::from_be_bytes(bytes[offset..offset + 4].try_into().ok()?) as usize;
        let chunk_type = &bytes[offset + 4..offset + 8];
        let data_start = offset + 8;
        let data_end = data_start.checked_add(length)?;
        if data_end.checked_add(4)? > bytes.len() {
            return None;
        }

        if chunk_type == b"pHYs" && length >= 9 {
            let x_ppu = u32::from_be_bytes(bytes[data_start..data_start + 4].try_into().ok()?);
            let y_ppu = u32::from_be_bytes(bytes[data_start + 4..data_start + 8].try_into().ok()?);
            let unit = bytes[data_start + 8];
            if unit == 1 && x_ppu > 0 && y_ppu > 0 {
                return Some((x_ppu as f64 * 0.0254, y_ppu as f64 * 0.0254, true));
            }
            return None;
        }

        offset = data_end + 4;
    }

    None
}

fn extract_dpi_from_jpeg_jfif(bytes: &[u8]) -> Option<(f64, f64, bool)> {
    if bytes.len() < 4 || !bytes.starts_with(&[0xFF, 0xD8]) {
        return None;
    }

    let mut offset = 2usize;
    while offset + 4 <= bytes.len() {
        if bytes[offset] != 0xFF {
            return None;
        }
        while offset < bytes.len() && bytes[offset] == 0xFF {
            offset += 1;
        }
        if offset >= bytes.len() {
            return None;
        }
        let marker = bytes[offset];
        offset += 1;
        if marker == 0xDA || marker == 0xD9 {
            break;
        }
        if offset + 2 > bytes.len() {
            return None;
        }
        let segment_len = u16::from_be_bytes([bytes[offset], bytes[offset + 1]]) as usize;
        if segment_len < 2 || offset + segment_len > bytes.len() {
            return None;
        }
        let data_start = offset + 2;
        let data_end = offset + segment_len;
        if marker == 0xE0
            && data_end >= data_start + 14
            && &bytes[data_start..data_start + 5] == b"JFIF\0"
        {
            let unit = bytes[data_start + 7];
            let x_density = u16::from_be_bytes([bytes[data_start + 8], bytes[data_start + 9]]);
            let y_density = u16::from_be_bytes([bytes[data_start + 10], bytes[data_start + 11]]);
            if x_density == 0 || y_density == 0 {
                return None;
            }
            return match unit {
                1 => Some((x_density as f64, y_density as f64, true)),
                2 => Some((x_density as f64 * 2.54, y_density as f64 * 2.54, true)),
                _ => None,
            };
        }
        offset = data_end;
    }

    None
}

fn extract_dpi_from_tiff(bytes: &[u8]) -> Option<(f64, f64, bool)> {
    if bytes.len() < 8 {
        return None;
    }

    let little = match &bytes[..2] {
        b"II" => true,
        b"MM" => false,
        _ => return None,
    };
    if read_u16(bytes, 2, little)? != 42 {
        return None;
    }
    let ifd_offset = read_u32(bytes, 4, little)? as usize;
    let entry_count = read_u16(bytes, ifd_offset, little)? as usize;

    let mut x_offset = None;
    let mut y_offset = None;
    let mut unit = 2u16;
    for i in 0..entry_count {
        let entry = ifd_offset + 2 + i * 12;
        if entry + 12 > bytes.len() {
            return None;
        }
        let tag = read_u16(bytes, entry, little)?;
        let field_type = read_u16(bytes, entry + 2, little)?;
        let count = read_u32(bytes, entry + 4, little)?;
        let value = read_u32(bytes, entry + 8, little)?;
        match tag {
            282 if field_type == 5 && count == 1 => x_offset = Some(value as usize),
            283 if field_type == 5 && count == 1 => y_offset = Some(value as usize),
            296 if field_type == 3 && count == 1 => {
                unit = if little {
                    (value & 0xffff) as u16
                } else {
                    (value >> 16) as u16
                };
            }
            _ => {}
        }
    }

    let x = read_rational(bytes, x_offset?, little)?;
    let y = read_rational(bytes, y_offset?, little)?;
    if x <= 0.0 || y <= 0.0 {
        return None;
    }
    match unit {
        2 => Some((x, y, true)),
        3 => Some((x * 2.54, y * 2.54, true)),
        _ => Some((x, y, false)),
    }
}

fn read_u16(bytes: &[u8], offset: usize, little: bool) -> Option<u16> {
    let raw: [u8; 2] = bytes.get(offset..offset + 2)?.try_into().ok()?;
    Some(if little {
        u16::from_le_bytes(raw)
    } else {
        u16::from_be_bytes(raw)
    })
}

fn read_u32(bytes: &[u8], offset: usize, little: bool) -> Option<u32> {
    let raw: [u8; 4] = bytes.get(offset..offset + 4)?.try_into().ok()?;
    Some(if little {
        u32::from_le_bytes(raw)
    } else {
        u32::from_be_bytes(raw)
    })
}

fn read_rational(bytes: &[u8], offset: usize, little: bool) -> Option<f64> {
    let numerator = read_u32(bytes, offset, little)?;
    let denominator = read_u32(bytes, offset + 4, little)?;
    if denominator == 0 {
        None
    } else {
        Some(numerator as f64 / denominator as f64)
    }
}

/// 从 EXIF 数据提取 DPI 信息
fn extract_dpi_from_exif(bytes: &[u8]) -> Option<(f64, f64)> {
    let reader = ExifReader::new();
    let mut cursor = Cursor::new(bytes);

    let exif_data = match reader.read_from_container(&mut cursor) {
        Ok(data) => data,
        Err(_) => return None,
    };

    let mut x_res = None;
    let mut y_res = None;
    let mut unit = None;

    for field in exif_data.fields() {
        match field.tag {
            exif::Tag::XResolution => {
                if let exif::Value::Rational(ref r) = field.value {
                    if !r.is_empty() {
                        x_res = Some(r[0].to_f64());
                    }
                }
            }
            exif::Tag::YResolution => {
                if let exif::Value::Rational(ref r) = field.value {
                    if !r.is_empty() {
                        y_res = Some(r[0].to_f64());
                    }
                }
            }
            exif::Tag::ResolutionUnit => {
                if let exif::Value::Short(ref s) = field.value {
                    if !s.is_empty() {
                        unit = Some(s[0]);
                    }
                }
            }
            _ => {}
        }
    }

    match (x_res, y_res, unit) {
        (Some(x), Some(y), Some(u)) => {
            // 单位转换: 2=英寸, 3=厘米
            let dpi_multiplier = match u {
                2 => 1.0,  // 每英寸
                3 => 2.54, // 每厘米 → 转英寸
                _ => 1.0,  // 未知单位，默认英寸
            };
            Some((x * dpi_multiplier, y * dpi_multiplier))
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{DynamicImage, ImageBuffer, Rgba};
    use std::io::Cursor;

    fn make_test_png() -> Vec<u8> {
        let img = ImageBuffer::from_fn(8, 8, |x, y| {
            if x == y {
                Rgba([0, 0, 0, 255])
            } else {
                Rgba([255, 255, 255, 255])
            }
        });
        let mut cursor = Cursor::new(Vec::new());
        DynamicImage::ImageRgba8(img)
            .write_to(&mut cursor, image::ImageFormat::Png)
            .unwrap();
        cursor.into_inner()
    }

    fn make_png_phys_chunk(dpi: f64) -> Vec<u8> {
        let ppm = (dpi / 0.0254).round() as u32;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"\x89PNG\r\n\x1a\n");
        bytes.extend_from_slice(&9u32.to_be_bytes());
        bytes.extend_from_slice(b"pHYs");
        bytes.extend_from_slice(&ppm.to_be_bytes());
        bytes.extend_from_slice(&ppm.to_be_bytes());
        bytes.push(1);
        bytes.extend_from_slice(&0u32.to_be_bytes());
        bytes
    }

    fn make_jpeg_jfif_header(dpi: u16) -> Vec<u8> {
        vec![
            0xFF,
            0xD8,
            0xFF,
            0xE0,
            0x00,
            0x10,
            b'J',
            b'F',
            b'I',
            b'F',
            0x00,
            0x01,
            0x02,
            0x01,
            (dpi >> 8) as u8,
            dpi as u8,
            (dpi >> 8) as u8,
            dpi as u8,
            0x00,
            0x00,
            0xFF,
            0xD9,
        ]
    }

    fn make_tiff_resolution(dpi: u32) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"II");
        bytes.extend_from_slice(&42u16.to_le_bytes());
        bytes.extend_from_slice(&8u32.to_le_bytes());
        bytes.extend_from_slice(&3u16.to_le_bytes());

        bytes.extend_from_slice(&282u16.to_le_bytes());
        bytes.extend_from_slice(&5u16.to_le_bytes());
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&50u32.to_le_bytes());

        bytes.extend_from_slice(&283u16.to_le_bytes());
        bytes.extend_from_slice(&5u16.to_le_bytes());
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&58u32.to_le_bytes());

        bytes.extend_from_slice(&296u16.to_le_bytes());
        bytes.extend_from_slice(&3u16.to_le_bytes());
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&2u16.to_le_bytes());
        bytes.extend_from_slice(&0u16.to_le_bytes());

        bytes.extend_from_slice(&0u32.to_le_bytes());
        bytes.extend_from_slice(&dpi.to_le_bytes());
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&dpi.to_le_bytes());
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes
    }

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
    fn test_from_bytes_valid_png() {
        let png = make_test_png();
        let (image, info) = RasterLoader::from_bytes(&png, Some("png")).unwrap();

        assert_eq!(image.dimensions(), (8, 8));
        assert_eq!(info.width, 8);
        assert_eq!(info.height, 8);
        assert_eq!(info.format, RasterFormat::Png);
    }

    #[test]
    fn test_from_file_valid_png() {
        let png = make_test_png();
        let path = std::env::temp_dir().join(format!(
            "raster_loader_test_{}_valid.png",
            std::process::id()
        ));
        std::fs::write(&path, &png).unwrap();

        let (image, info) = RasterLoader::from_file(&path).unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(image.dimensions(), (8, 8));
        assert_eq!(info.width, 8);
        assert_eq!(info.height, 8);
        assert_eq!(info.format, RasterFormat::Png);
        assert_eq!(info.source_path, Some(path.display().to_string()));
    }

    #[test]
    fn test_from_file_and_from_bytes_consistent() {
        let png = make_test_png();
        let path = std::env::temp_dir().join(format!(
            "raster_loader_test_{}_consistent.png",
            std::process::id()
        ));
        std::fs::write(&path, &png).unwrap();

        let (_, file_info) = RasterLoader::from_file(&path).unwrap();
        let (_, bytes_info) = RasterLoader::from_bytes(&png, Some("consistent.png")).unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(file_info.width, bytes_info.width);
        assert_eq!(file_info.height, bytes_info.height);
        assert_eq!(file_info.format, bytes_info.format);
    }

    #[test]
    fn test_from_file_too_large() {
        let path = std::env::temp_dir().join(format!(
            "raster_loader_test_{}_too_large.png",
            std::process::id()
        ));
        let file = std::fs::File::create(&path).unwrap();
        file.set_len((100 * 1024 * 1024 + 1) as u64).unwrap();
        drop(file);

        let result = RasterLoader::from_file(&path);
        std::fs::remove_file(&path).ok();

        assert!(matches!(result, Err(RasterError::FileTooLarge { .. })));
    }

    #[test]
    fn test_raster_image_info() {
        let info = RasterImageInfo::new(800, 600, RasterFormat::Png);
        assert_eq!(info.width, 800);
        assert_eq!(info.height, 600);
        assert_eq!(info.dpi_x, None);
        assert_eq!(info.dpi_source, None);
        assert!(!info.dpi_trusted);
        assert_eq!(info.format, RasterFormat::Png);
    }

    #[test]
    fn test_png_phys_dpi_300() {
        let bytes = make_png_phys_chunk(300.0);
        let (x, y, trusted) = extract_dpi_from_png_phys(&bytes).unwrap();
        assert!((x - 300.0).abs() < 0.05);
        assert!((y - 300.0).abs() < 0.05);
        assert!(trusted);
    }

    #[test]
    fn test_jpeg_jfif_dpi_72() {
        let bytes = make_jpeg_jfif_header(72);
        let (x, y, trusted) = extract_dpi_from_jpeg_jfif(&bytes).unwrap();
        assert_eq!(x, 72.0);
        assert_eq!(y, 72.0);
        assert!(trusted);
    }

    #[test]
    fn test_tiff_resolution_dpi_96() {
        let bytes = make_tiff_resolution(96);
        let (x, y, trusted) = extract_dpi_from_tiff(&bytes).unwrap();
        assert_eq!(x, 96.0);
        assert_eq!(y, 96.0);
        assert!(trusted);
    }
}
