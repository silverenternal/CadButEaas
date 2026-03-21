//! wgpu 上下文管理

use log::info;

/// wgpu 上下文
///
/// 封装 wgpu 实例、设备和队列
pub struct WgpuContext {
    pub instance: wgpu::Instance,
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

impl WgpuContext {
    /// 创建 wgpu 上下文
    pub async fn new() -> Result<Self, String> {
        info!("初始化 wgpu 上下文...");

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .await
            .ok_or_else(|| "无法获取 wgpu 适配器".to_string())?;

        let adapter_info = adapter.get_info();
        info!("wgpu 适配器：{} ({:?})", adapter_info.name, adapter_info.backend);

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("CAD Accelerator Device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                },
                None,
            )
            .await
            .map_err(|e| format!("无法创建 wgpu 设备：{}", e))?;

        info!("wgpu 上下文初始化成功");

        Ok(Self {
            instance,
            adapter,
            device,
            queue,
        })
    }

    /// 获取适配器信息
    pub fn adapter_info(&self) -> wgpu::AdapterInfo {
        self.adapter.get_info()
    }

    /// 检查是否支持计算着色器
    pub fn supports_compute(&self) -> bool {
        true // wgpu 始终支持计算着色器
    }

    /// 获取最大存储缓冲区大小
    pub fn max_storage_buffer_size(&self) -> u64 {
        self.adapter.limits().max_storage_buffer_binding_size as u64
    }

    /// 获取最大统一缓冲区大小
    pub fn max_uniform_buffer_size(&self) -> u64 {
        self.adapter.limits().max_uniform_buffer_binding_size as u64
    }
}

impl std::fmt::Debug for WgpuContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let info = self.adapter.get_info();
        f.debug_struct("WgpuContext")
            .field("adapter", &info.name)
            .field("backend", &info.backend)
            .finish()
    }
}
