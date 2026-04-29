//! 光栅图片加载器
//!
//! 统一负责 PNG/JPG/BMP/TIFF/WebP/PBM/PGM/PPM 格式图片的加载、格式检测和元数据提取，
//! 输出 `image::DynamicImage` 直接对接 `VectorizeService`。
//!
//! 额外功能：
//! - EXIF DPI 提取
//! - 扫描件预处理（阴影去除、对比度增强、去噪、锐化、二值化）

pub mod format_detect;
pub mod loader;
pub mod preprocess;

pub use format_detect::{detect_raster_format, RasterFormat};
pub use loader::{RasterImageInfo, RasterLoader};
pub use preprocess::{PreprocessConfig, RasterPreprocessor};
