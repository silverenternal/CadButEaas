//! 加速器 Trait 定义

use crate::error::Result;
use crate::types::{
    AcceleratorAvailability, AcceleratorCapabilities, AcceleratorOp, Arc as Arc2d, ArcFitConfig,
    ContourExtractConfig, Contours, EdgeDetectConfig, EdgeMap, Image, Point2, SnapConfig,
};
use async_trait::async_trait;

/// 加速器统一接口
///
/// 所有加速器后端（CPU、wgpu、CUDA、OpenCL）都必须实现此 trait
#[async_trait]
pub trait Accelerator: Send + Sync + std::fmt::Debug {
    /// 加速器名称（如 "CPU", "wgpu", "CUDA", "OpenCL"）
    fn name(&self) -> &str;

    /// 检查加速器可用性
    fn availability(&self) -> AcceleratorAvailability;

    /// 获取加速器能力描述
    fn capabilities(&self) -> AcceleratorCapabilities;

    /// 边缘检测
    ///
    /// # 参数
    /// * `image` - 输入图像
    /// * `config` - 边缘检测配置
    ///
    /// # 返回
    /// 边缘检测结果
    async fn edge_detect(&self, image: &Image, config: &EdgeDetectConfig) -> Result<EdgeMap>;

    /// 轮廓提取
    ///
    /// # 参数
    /// * `edges` - 边缘图
    /// * `config` - 轮廓提取配置
    ///
    /// # 返回
    /// 轮廓列表
    async fn contour_extract(
        &self,
        edges: &EdgeMap,
        config: &ContourExtractConfig,
    ) -> Result<Contours>;

    /// 圆弧拟合
    ///
    /// # 参数
    /// * `points` - 点集
    /// * `config` - 圆弧拟合配置
    ///
    /// # 返回
    /// 拟合的圆弧
    async fn arc_fit(&self, points: &[Point2], config: &ArcFitConfig) -> Result<Arc2d>;

    /// 端点吸附
    ///
    /// # 参数
    /// * `points` - 点集
    /// * `config` - 吸附配置
    ///
    /// # 返回
    /// 吸附后的点集
    async fn snap_endpoints(&self, points: &[Point2], config: &SnapConfig) -> Result<Vec<Point2>>;

    /// 检查是否支持指定操作
    fn supports_op(&self, op: AcceleratorOp) -> bool {
        self.availability().supports(op)
    }

    /// 获取性能评分（相对 CPU 的倍数）
    fn performance_score(&self) -> f32 {
        self.capabilities().performance_score
    }
}

/// 加速器引用类型（用于运行时多态）
pub type AcceleratorRef = Box<dyn Accelerator>;

/// 可选的加速器引用（用于回退场景）
pub type OptionAcceleratorRef = Option<Box<dyn Accelerator>>;
