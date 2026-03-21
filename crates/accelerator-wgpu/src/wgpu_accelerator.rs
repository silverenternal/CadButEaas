//! wgpu 加速器实现

use accelerator_api::{
    Accelerator, AcceleratorAvailability, AcceleratorCapabilities,
    Arc as AcceleratorArc, ArcFitConfig, Contours, ContourExtractConfig, EdgeMap, Image,
    Point2, Precision, SnapConfig, EdgeDetectConfig,
    AcceleratorResult,
};
use async_trait::async_trait;
use log::{info, debug};

use crate::context::WgpuContext;

/// wgpu 加速器
///
/// 使用 WebGPU 计算着色器加速几何处理
#[derive(Debug)]
pub struct WgpuAccelerator {
    context: WgpuContext,
    capabilities: AcceleratorCapabilities,
}

impl WgpuAccelerator {
    /// 创建新的 wgpu 加速器
    pub async fn new() -> Result<Self, String> {
        let context = WgpuContext::new().await?;
        
        let adapter_info = context.adapter_info();
        
        let capabilities = AcceleratorCapabilities {
            name: format!("wgpu ({})", adapter_info.name),
            memory_bandwidth_gbps: Self::estimate_bandwidth(&adapter_info),
            compute_units: Self::estimate_compute_units(&adapter_info),
            max_memory_mb: context.max_storage_buffer_size() / (1024 * 1024),
            supported_precision: vec![Precision::F32, Precision::F64],
            performance_score: Self::estimate_performance_score(&adapter_info),
            supports_async: true,
            supports_concurrent: true,
        };

        info!("wgpu 加速器已创建：{}", capabilities.name);

        Ok(Self {
            context,
            capabilities,
        })
    }

    /// 同步创建（使用 pollster）
    pub fn new_sync() -> Result<Self, String> {
        pollster::block_on(Self::new())
    }

    /// 估算内存带宽
    fn estimate_bandwidth(info: &wgpu::AdapterInfo) -> f32 {
        // 根据后端类型估算
        match info.backend {
            wgpu::Backend::Vulkan | wgpu::Backend::Dx12 | wgpu::Backend::Metal => {
                // 独立显卡：200-1000 GB/s
                400.0
            }
            wgpu::Backend::Gl | wgpu::Backend::BrowserWebGpu => {
                // 集成显卡/OpenGL：50-100 GB/s
                60.0
            }
            _ => 100.0,
        }
    }

    /// 估算计算单元数量
    fn estimate_compute_units(info: &wgpu::AdapterInfo) -> u32 {
        // 根据后端类型估算
        match info.backend {
            wgpu::Backend::Vulkan | wgpu::Backend::Dx12 | wgpu::Backend::Metal => {
                // 独立显卡：1000-10000+
                2000
            }
            wgpu::Backend::Gl | wgpu::Backend::BrowserWebGpu => {
                // 集成显卡：100-500
                200
            }
            _ => 500,
        }
    }

    /// 估算性能评分（相对 CPU）
    fn estimate_performance_score(info: &wgpu::AdapterInfo) -> f32 {
        match info.backend {
            wgpu::Backend::Vulkan | wgpu::Backend::Dx12 | wgpu::Backend::Metal => {
                // 独立显卡：10-50x
                20.0
            }
            wgpu::Backend::Gl | wgpu::Backend::BrowserWebGpu => {
                // 集成显卡：2-10x
                5.0
            }
            _ => 8.0,
        }
    }
}

#[async_trait]
impl Accelerator for WgpuAccelerator {
    fn name(&self) -> &str {
        "wgpu"
    }

    fn availability(&self) -> AcceleratorAvailability {
        // 如果实例已创建，说明 wgpu 可用
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
        debug!("wgpu 边缘检测：{}x{}", image.width, image.height);

        // TODO: 实现 GPU 边缘检测
        // 目前回退到 CPU 实现
        // 未来实现：
        // 1. 将图像数据上传到 GPU 缓冲区
        // 2. 绑定计算着色器（Sobel/Canny）
        // 3. 调度并行计算
        // 4. 下载结果

        // 临时回退到 CPU 实现
        use accelerator_cpu::detect_edges_cpu;
        detect_edges_cpu(image, config)
    }

    async fn contour_extract(
        &self,
        edges: &EdgeMap,
        config: &ContourExtractConfig,
    ) -> AcceleratorResult<Contours> {
        debug!("wgpu 轮廓提取：{}x{}", edges.width, edges.height);

        // TODO: 实现 GPU 轮廓提取
        // 目前回退到 CPU 实现

        use accelerator_cpu::extract_contours_cpu;
        extract_contours_cpu(edges, config)
    }

    async fn arc_fit(&self, points: &[Point2], config: &ArcFitConfig) -> AcceleratorResult<AcceleratorArc> {
        debug!("wgpu 圆弧拟合：{} 个点", points.len());

        // TODO: 实现 GPU 圆弧拟合
        // 目前回退到 CPU 实现

        use accelerator_cpu::fit_arc_cpu;
        fit_arc_cpu(points, config)
    }

    async fn snap_endpoints(&self, points: &[Point2], config: &SnapConfig) -> AcceleratorResult<Vec<Point2>> {
        debug!("wgpu 端点吸附：{} 个点", points.len());

        // TODO: 实现 GPU 端点吸附
        // 目前回退到 CPU 实现

        use accelerator_cpu::snap_endpoints_cpu;
        snap_endpoints_cpu(points, config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore] // 需要 GPU 支持
    fn test_wgpu_creation() {
        let accelerator = WgpuAccelerator::new_sync();
        if let Ok(acc) = accelerator {
            assert_eq!(acc.name(), "wgpu");
            assert!(acc.availability().is_available());
        }
    }
}
