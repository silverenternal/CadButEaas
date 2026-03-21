//! CPU 加速器实现

use accelerator_api::{
    Accelerator, AcceleratorAvailability, AcceleratorCapabilities,
    Arc as AcceleratorArc, ArcFitConfig, Contours, ContourExtractConfig, EdgeMap, Image,
    Point2, Precision, SnapConfig, EdgeDetectConfig,
    AcceleratorResult,
};
use async_trait::async_trait;
use log::debug;

use crate::detect_edges_cpu;
use crate::extract_contours_cpu;
use crate::fit_arc_cpu;
use crate::snap_endpoints_cpu;

/// CPU 加速器
///
/// 纯 Rust 实现，使用 rayon 并行化加速
#[derive(Debug)]
pub struct CpuAccelerator {
    capabilities: AcceleratorCapabilities,
}

impl CpuAccelerator {
    /// 创建新的 CPU 加速器
    pub fn new() -> Self {
        Self {
            capabilities: AcceleratorCapabilities {
                name: "CPU".to_string(),
                memory_bandwidth_gbps: 50.0, // 估计值
                compute_units: num_cpus::get() as u32,
                max_memory_mb: Self::get_available_memory(),
                supported_precision: vec![Precision::F32, Precision::F64],
                performance_score: 1.0, // 基准
                supports_async: true,
                supports_concurrent: true,
            },
        }
    }

    /// 获取可用内存（MB）
    fn get_available_memory() -> u64 {
        // 简单估计：系统内存的 50%
        // 在实际生产中应该使用 sysinfo 等 crate 获取真实值
        8192 // 默认 8GB
    }
}

impl Default for CpuAccelerator {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Accelerator for CpuAccelerator {
    fn name(&self) -> &str {
        "CPU"
    }

    fn availability(&self) -> AcceleratorAvailability {
        // CPU 总是可用
        AcceleratorAvailability::Available
    }

    fn capabilities(&self) -> AcceleratorCapabilities {
        self.capabilities.clone()
    }

    async fn edge_detect(
        &self,
        image: &Image,
        config: &EdgeDetectConfig,
    ) -> AcceleratorResult<EdgeMap> {
        debug!("CPU 边缘检测：{}x{}", image.width, image.height);
        let result = detect_edges_cpu(image, config);
        debug!("CPU 边缘检测完成");
        result
    }

    async fn contour_extract(
        &self,
        edges: &EdgeMap,
        config: &ContourExtractConfig,
    ) -> AcceleratorResult<Contours> {
        debug!("CPU 轮廓提取：{}x{}", edges.width, edges.height);
        let result = extract_contours_cpu(edges, config);
        debug!("CPU 轮廓提取完成");
        result
    }

    async fn arc_fit(&self, points: &[Point2], config: &ArcFitConfig) -> AcceleratorResult<AcceleratorArc> {
        debug!("CPU 圆弧拟合：{} 个点", points.len());
        let result = fit_arc_cpu(points, config);
        debug!("CPU 圆弧拟合完成");
        result
    }

    async fn snap_endpoints(&self, points: &[Point2], config: &SnapConfig) -> AcceleratorResult<Vec<Point2>> {
        debug!("CPU 端点吸附：{} 个点", points.len());
        let result = snap_endpoints_cpu(points, config);
        debug!("CPU 端点吸附完成");
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_accelerator_creation() {
        let accelerator = CpuAccelerator::new();
        assert_eq!(accelerator.name(), "CPU");
        assert!(accelerator.availability().is_available());
    }

    #[test]
    fn test_cpu_capabilities() {
        let accelerator = CpuAccelerator::new();
        let caps = accelerator.capabilities();
        assert_eq!(caps.name, "CPU");
        assert!(caps.compute_units > 0);
        assert!(caps.max_memory_mb > 0);
    }
}
