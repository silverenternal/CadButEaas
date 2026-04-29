//! 光栅图片格式检测
//!
//! 通过魔数（magic bytes）和文件扩展名检测 PNG/JPG/BMP/TIFF/WebP 格式。

use std::path::Path;

/// 支持的光栅图片格式
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RasterFormat {
    Png,
    Jpeg,
    Bmp,
    Tiff,
    WebP,
    /// Netpbm Portable Bitmap (P1/P4 - 单色)
    Pbm,
    /// Netpbm Portable Graymap (P2/P5 - 灰度)
    Pgm,
    /// Netpbm Portable Pixmap (P3/P6 - 彩色)
    Ppm,
    /// Graphics Interchange Format
    Gif,
    /// Icon 格式 (ICO)
    Ico,
    /// Farbfeld 无损格式
    Farbfeld,
    /// X11 Bitmap
    Xbm,
    /// Radiance RGBE (HDR)
    Hdr,
    /// Targa (TGA)
    Tga,
}

impl RasterFormat {
    /// 获取文件扩展名
    pub fn extension(&self) -> &'static str {
        match self {
            RasterFormat::Png => "png",
            RasterFormat::Jpeg => "jpg",
            RasterFormat::Bmp => "bmp",
            RasterFormat::Tiff => "tiff",
            RasterFormat::WebP => "webp",
            RasterFormat::Pbm => "pbm",
            RasterFormat::Pgm => "pgm",
            RasterFormat::Ppm => "ppm",
            RasterFormat::Gif => "gif",
            RasterFormat::Ico => "ico",
            RasterFormat::Farbfeld => "ff",
            RasterFormat::Xbm => "xbm",
            RasterFormat::Hdr => "hdr",
            RasterFormat::Tga => "tga",
        }
    }

    /// 判断是否为 Netpbm 格式家族（PBM/PGM/PPM）
    pub fn is_netpbm(&self) -> bool {
        matches!(
            self,
            RasterFormat::Pbm | RasterFormat::Pgm | RasterFormat::Ppm
        )
    }
}

/// 从字节数据检测光栅图片格式
///
/// # 参数
/// * `data` - 文件头字节（至少需要前 12 字节）
/// * `filename` - 可选的文件名，用于扩展名兜底判断
///
/// # 返回
/// * `Some(RasterFormat)` - 检测到的格式
/// * `None` - 无法识别
///
/// # 魔数参考
/// | 格式 | 魔数（十六进制） | 字节偏移 |
/// |------|-----------------|---------|
/// | PNG  | 89 50 4E 47     | 0-3     |
/// | JPEG | FF D8 FF        | 0-2     |
/// | BMP  | 42 4D           | 0-1     |
/// | TIFF (LE) | 49 49 2A 00 | 0-3   |
/// | TIFF (BE) | 4D 4D 00 2A | 0-3   |
/// | WebP | 52 49 46 46 .. 57 45 42 50 | 0-3, 8-11 |
pub fn detect_raster_format(data: &[u8], filename: Option<&str>) -> Option<RasterFormat> {
    // 优先通过扩展名判断（快速路径）
    if let Some(name) = filename {
        if let Some(ext) = Path::new(name).extension().and_then(|e| e.to_str()) {
            let ext_lower = ext.to_lowercase();
            let format = match ext_lower.as_str() {
                "png" => RasterFormat::Png,
                "jpg" | "jpeg" => RasterFormat::Jpeg,
                "bmp" => RasterFormat::Bmp,
                "tif" | "tiff" => RasterFormat::Tiff,
                "webp" => RasterFormat::WebP,
                "pbm" => RasterFormat::Pbm,
                "pgm" => RasterFormat::Pgm,
                "ppm" => RasterFormat::Ppm,
                "gif" => RasterFormat::Gif,
                "ico" => RasterFormat::Ico,
                "ff" => RasterFormat::Farbfeld,
                "xbm" => RasterFormat::Xbm,
                "hdr" | "pic" => RasterFormat::Hdr,
                "tga" => RasterFormat::Tga,
                _ => return detect_by_magic(data), // 扩展名不匹配，回退到魔数检测
            };
            return Some(format);
        }
    }

    // 通过魔数检测
    detect_by_magic(data)
}

/// 仅通过魔数检测格式
fn detect_by_magic(data: &[u8]) -> Option<RasterFormat> {
    // Netpbm 格式只需要 2 字节，先检查
    if data.len() >= 2 {
        match (data[0], data[1]) {
            (b'P', b'1') => return Some(RasterFormat::Pbm),
            (b'P', b'2') => return Some(RasterFormat::Pgm),
            (b'P', b'3') => return Some(RasterFormat::Ppm),
            (b'P', b'4') => return Some(RasterFormat::Pbm),
            (b'P', b'5') => return Some(RasterFormat::Pgm),
            (b'P', b'6') => return Some(RasterFormat::Ppm),
            _ => {}
        }
    }

    // 其他格式至少需要 4 字节
    if data.len() < 4 {
        return None;
    }

    // JPEG: FF D8 FF
    if data.starts_with(&[0xFF, 0xD8, 0xFF]) {
        return Some(RasterFormat::Jpeg);
    }

    // PNG: 89 50 4E 47
    if data.starts_with(&[0x89, 0x50, 0x4E, 0x47]) {
        return Some(RasterFormat::Png);
    }

    // BMP: 42 4D
    if data.starts_with(&[0x42, 0x4D]) {
        return Some(RasterFormat::Bmp);
    }

    // TIFF Little Endian: 49 49 2A 00
    if data.starts_with(&[0x49, 0x49, 0x2A, 0x00]) {
        return Some(RasterFormat::Tiff);
    }

    // TIFF Big Endian: 4D 4D 00 2A
    if data.starts_with(&[0x4D, 0x4D, 0x00, 0x2A]) {
        return Some(RasterFormat::Tiff);
    }

    // WebP: RIFF .... WEBP (前 4 字节 + 第 8-11 字节)
    if data.len() >= 12 && data.starts_with(b"RIFF") && &data[8..12] == b"WEBP" {
        return Some(RasterFormat::WebP);
    }

    // GIF: GIF87a 或 GIF89a
    if data.len() >= 6 && data.starts_with(b"GIF87") || data.starts_with(b"GIF89") {
        return Some(RasterFormat::Gif);
    }

    // ICO: 00 00 01 00
    if data.starts_with(&[0x00, 0x00, 0x01, 0x00]) {
        return Some(RasterFormat::Ico);
    }

    // Farbfeld: farbfeld magic (8 bytes)
    if data.len() >= 8 && data.starts_with(b"farbfeld") {
        return Some(RasterFormat::Farbfeld);
    }

    // XBM: #define 前缀 (X11 Bitmap)
    if data.len() >= 7 && data.starts_with(b"#define") {
        return Some(RasterFormat::Xbm);
    }

    // Radiance RGBE (HDR): #?RADIANCE 或 #?RGBE
    if data.len() >= 10 && (data.starts_with(b"#?RADIANCE") || data.starts_with(b"#?RGBE")) {
        return Some(RasterFormat::Hdr);
    }

    // TGA: 没有标准魔数，但可以通过扩展名检测

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_png_magic() {
        let data = [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A];
        assert_eq!(detect_raster_format(&data, None), Some(RasterFormat::Png));
    }

    #[test]
    fn test_detect_jpeg_magic() {
        let data = [0xFF, 0xD8, 0xFF, 0xE0];
        assert_eq!(detect_raster_format(&data, None), Some(RasterFormat::Jpeg));
    }

    #[test]
    fn test_detect_bmp_magic() {
        let data = [0x42, 0x4D, 0x00, 0x00];
        assert_eq!(detect_raster_format(&data, None), Some(RasterFormat::Bmp));
    }

    #[test]
    fn test_detect_tiff_le_magic() {
        let data = [0x49, 0x49, 0x2A, 0x00];
        assert_eq!(detect_raster_format(&data, None), Some(RasterFormat::Tiff));
    }

    #[test]
    fn test_detect_tiff_be_magic() {
        let data = [0x4D, 0x4D, 0x00, 0x2A];
        assert_eq!(detect_raster_format(&data, None), Some(RasterFormat::Tiff));
    }

    #[test]
    fn test_detect_webp_magic() {
        let mut data = vec![0u8; 12];
        data[0..4].copy_from_slice(b"RIFF");
        data[4..8].copy_from_slice(&[0x00, 0x00, 0x00, 0x00]); // size placeholder
        data[8..12].copy_from_slice(b"WEBP");
        assert_eq!(detect_raster_format(&data, None), Some(RasterFormat::WebP));
    }

    #[test]
    fn test_detect_by_extension() {
        let data = b"not_an_image_header";
        assert_eq!(
            detect_raster_format(data, Some("test.PNG")),
            Some(RasterFormat::Png)
        );
        assert_eq!(
            detect_raster_format(data, Some("photo.jpg")),
            Some(RasterFormat::Jpeg)
        );
        assert_eq!(
            detect_raster_format(data, Some("scan.tiff")),
            Some(RasterFormat::Tiff)
        );
        assert_eq!(
            detect_raster_format(data, Some("image.webp")),
            Some(RasterFormat::WebP)
        );
    }

    #[test]
    fn test_detect_unknown_format() {
        let data = b"unknown format data here";
        assert_eq!(detect_raster_format(data, None), None);
        assert_eq!(detect_raster_format(data, Some("file.xyz")), None);
    }

    #[test]
    fn test_magic_overrides_extension() {
        // 扩展名为 .txt 但实际是 PNG 数据
        let data = [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A];
        assert_eq!(
            detect_raster_format(&data, Some("misleading.txt")),
            Some(RasterFormat::Png)
        );
    }

    #[test]
    fn test_short_data_returns_none() {
        let data = [0x89];
        assert_eq!(detect_raster_format(&data, None), None);
    }

    #[test]
    fn test_detect_netpbm_magic() {
        // P1 (ASCII PBM)
        assert_eq!(detect_raster_format(b"P1", None), Some(RasterFormat::Pbm));
        // P2 (ASCII PGM)
        assert_eq!(detect_raster_format(b"P2", None), Some(RasterFormat::Pgm));
        // P3 (ASCII PPM)
        assert_eq!(detect_raster_format(b"P3", None), Some(RasterFormat::Ppm));
        // P4 (Binary PBM)
        assert_eq!(detect_raster_format(b"P4", None), Some(RasterFormat::Pbm));
        // P5 (Binary PGM)
        assert_eq!(detect_raster_format(b"P5", None), Some(RasterFormat::Pgm));
        // P6 (Binary PPM)
        assert_eq!(detect_raster_format(b"P6", None), Some(RasterFormat::Ppm));
    }

    #[test]
    fn test_detect_netpbm_by_extension() {
        let data = b"not_an_image_header";
        assert_eq!(
            detect_raster_format(data, Some("test.pbm")),
            Some(RasterFormat::Pbm)
        );
        assert_eq!(
            detect_raster_format(data, Some("image.pgm")),
            Some(RasterFormat::Pgm)
        );
        assert_eq!(
            detect_raster_format(data, Some("photo.ppm")),
            Some(RasterFormat::Ppm)
        );
    }

    #[test]
    fn test_is_netpbm() {
        assert!(RasterFormat::Pbm.is_netpbm());
        assert!(RasterFormat::Pgm.is_netpbm());
        assert!(RasterFormat::Ppm.is_netpbm());
        assert!(!RasterFormat::Png.is_netpbm());
        assert!(!RasterFormat::Jpeg.is_netpbm());
        assert!(!RasterFormat::Bmp.is_netpbm());
        assert!(!RasterFormat::Tiff.is_netpbm());
        assert!(!RasterFormat::WebP.is_netpbm());
    }

    #[test]
    fn test_extension() {
        assert_eq!(RasterFormat::Pbm.extension(), "pbm");
        assert_eq!(RasterFormat::Pgm.extension(), "pgm");
        assert_eq!(RasterFormat::Ppm.extension(), "ppm");
    }
}
