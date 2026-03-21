//! 加速器相关的数据类型

use serde::{Deserialize, Serialize};

/// 加速器可用性状态
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AcceleratorAvailability {
    /// 完全可用
    Available,
    /// 不可用，附带原因
    Unavailable(String),
    /// 部分可用（某些操作不支持）
    Partial {
        /// 支持的操作列表
        supported_ops: Vec<AcceleratorOp>,
        /// 不支持的操作列表
        unsupported_ops: Vec<AcceleratorOp>,
    },
}

impl AcceleratorAvailability {
    /// 检查是否支持指定操作
    pub fn supports(&self, op: AcceleratorOp) -> bool {
        match self {
            Self::Available => true,
            Self::Unavailable(_) => false,
            Self::Partial { supported_ops, .. } => supported_ops.contains(&op),
        }
    }

    /// 是否完全可用
    pub fn is_available(&self) -> bool {
        matches!(self, Self::Available)
    }
}

/// 加速器能力描述
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AcceleratorCapabilities {
    /// 加速器名称
    pub name: String,
    /// 内存带宽 (GB/s)
    pub memory_bandwidth_gbps: f32,
    /// 计算单元数量
    pub compute_units: u32,
    /// 最大可用内存 (MB)
    pub max_memory_mb: u64,
    /// 支持的精度类型
    pub supported_precision: Vec<Precision>,
    /// 性能评分（相对 CPU 的倍数，1.0 = 与 CPU 同性能）
    pub performance_score: f32,
    /// 是否支持异步执行
    pub supports_async: bool,
    /// 是否支持并发执行
    pub supports_concurrent: bool,
}

impl Default for AcceleratorCapabilities {
    fn default() -> Self {
        Self {
            name: "Unknown".to_string(),
            memory_bandwidth_gbps: 0.0,
            compute_units: 0,
            max_memory_mb: 0,
            supported_precision: vec![Precision::F32],
            performance_score: 1.0,
            supports_async: false,
            supports_concurrent: false,
        }
    }
}

/// 数值精度类型
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Precision {
    /// 32 位浮点
    F32,
    /// 64 位浮点
    F64,
    /// 16 位浮点（半精度）
    F16,
    /// 8 位整数
    I8,
    /// 32 位整数
    I32,
}

/// 加速器支持的操作类型
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AcceleratorOp {
    /// 边缘检测
    EdgeDetect,
    /// 轮廓提取
    ContourExtract,
    /// 圆弧拟合
    ArcFit,
    /// R*-tree 构建
    RTreeBuild,
    /// 端点吸附
    EndpointSnap,
    /// 交点检测
    IntersectionDetect,
    /// 多边形简化
    PolygonSimplify,
    /// 阈值处理
    Threshold,
    /// 骨架化
    Skeletonize,
}

impl AcceleratorOp {
    /// 获取操作的人类可读名称
    pub fn name(&self) -> &'static str {
        match self {
            Self::EdgeDetect => "边缘检测",
            Self::ContourExtract => "轮廓提取",
            Self::ArcFit => "圆弧拟合",
            Self::RTreeBuild => "R*-tree 构建",
            Self::EndpointSnap => "端点吸附",
            Self::IntersectionDetect => "交点检测",
            Self::PolygonSimplify => "多边形简化",
            Self::Threshold => "阈值处理",
            Self::Skeletonize => "骨架化",
        }
    }
}

/// 图像数据（加速器输入）
#[derive(Debug, Clone)]
pub struct Image {
    /// 图像宽度
    pub width: u32,
    /// 图像高度
    pub height: u32,
    /// 像素数据（灰度，0-255）
    pub data: Vec<u8>,
}

impl Image {
    /// 从灰度图像创建
    pub fn from_gray(image: &image::GrayImage) -> Self {
        let (width, height) = image.dimensions();
        Self {
            width,
            height,
            data: image.as_raw().clone(),
        }
    }

    /// 转换为灰度图像
    pub fn to_gray(&self) -> image::GrayImage {
        image::GrayImage::from_raw(self.width, self.height, self.data.clone())
            .unwrap_or_else(|| image::GrayImage::new(self.width, self.height))
    }
}

/// 边缘检测结果
#[derive(Debug, Clone)]
pub struct EdgeMap {
    /// 图像宽度
    pub width: u32,
    /// 图像高度
    pub height: u32,
    /// 边缘像素数据（0 = 边缘，255 = 非边缘）
    pub data: Vec<u8>,
}

impl EdgeMap {
    /// 创建空的边缘图
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            data: vec![255u8; (width * height) as usize],
        }
    }

    /// 从灰度图像创建（0 = 边缘，255 = 非边缘）
    pub fn from_gray(image: &image::GrayImage) -> Self {
        let (width, height) = image.dimensions();
        Self {
            width,
            height,
            data: image.as_raw().clone(),
        }
    }

    /// 转换为灰度图像
    pub fn to_gray(&self) -> image::GrayImage {
        image::GrayImage::from_raw(self.width, self.height, self.data.clone())
            .unwrap_or_else(|| image::GrayImage::new(self.width, self.height))
    }
}

/// 轮廓数据
pub type Contours = Vec<common_types::Polyline>;

/// 2D 点
pub type Point2 = common_types::Point2;

/// 圆弧
#[derive(Debug, Clone)]
pub struct Arc {
    /// 圆心
    pub center: Point2,
    /// 半径
    pub radius: f64,
    /// 起始角度（弧度）
    pub start_angle: f64,
    /// 终止角度（弧度）
    pub end_angle: f64,
}

impl Arc {
    /// 创建圆弧
    pub fn new(center: Point2, radius: f64, start_angle: f64, end_angle: f64) -> Self {
        Self {
            center,
            radius,
            start_angle,
            end_angle,
        }
    }
}

/// 边缘检测配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeDetectConfig {
    /// Canny 低阈值
    pub low_threshold: f64,
    /// Canny 高阈值
    pub high_threshold: f64,
    /// Sobel 算子大小
    pub sobel_kernel_size: u32,
    /// 是否使用自适应阈值
    pub adaptive_threshold: bool,
}

impl Default for EdgeDetectConfig {
    fn default() -> Self {
        Self {
            low_threshold: 50.0,
            high_threshold: 150.0,
            sobel_kernel_size: 3,
            adaptive_threshold: true,
        }
    }
}

/// 轮廓提取配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContourExtractConfig {
    /// 最小轮廓长度（点数）
    pub min_contour_length: usize,
    /// 简化精度（Douglas-Peucker epsilon）
    pub simplify_epsilon: f64,
    /// 是否简化轮廓
    pub simplify: bool,
}

impl Default for ContourExtractConfig {
    fn default() -> Self {
        Self {
            min_contour_length: 3,
            simplify_epsilon: 0.5,
            simplify: true,
        }
    }
}

/// 圆弧拟合配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArcFitConfig {
    /// 最大拟合误差
    pub max_error: f64,
    /// 最小点数
    pub min_points: usize,
}

impl Default for ArcFitConfig {
    fn default() -> Self {
        Self {
            max_error: 0.1,
            min_points: 3,
        }
    }
}

/// 端点吸附配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SnapConfig {
    /// 吸附容差
    pub tolerance: f64,
    /// 是否构建 R*-tree 索引
    pub use_rtree: bool,
}

impl Default for SnapConfig {
    fn default() -> Self {
        Self {
            tolerance: 0.01,
            use_rtree: true,
        }
    }
}
