//! 光栅图片加载器
//!
//! 统一负责 PNG/JPG/BMP/TIFF/WebP 格式图片的加载、格式检测和元数据提取，
//! 输出 `image::DynamicImage` 直接对接 `VectorizeService`。

pub mod format_detect;
pub mod loader;

pub use format_detect::{detect_raster_format, RasterFormat};
pub use loader::{RasterImageInfo, RasterLoader};
