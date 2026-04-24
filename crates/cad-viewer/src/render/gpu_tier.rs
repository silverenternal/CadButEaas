//! GPU 分级降级系统
//!
//! 设计目标：
//! - 自动检测 GPU 能力等级
//! - 根据等级调整视觉效果质量
//! - 保证低端设备流畅运行
//! - 高端设备提供最佳体验

use std::fmt;

/// GPU 性能等级
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GpuTier {
    /// 未知（未检测）
    #[default]
    Unknown,
    /// 低端（集成显卡/老旧 GPU）
    /// - 禁用毛玻璃效果
    /// - 禁用 MSAA
    /// - 降低阴影质量
    Low,
    /// 中端（主流独立显卡）
    /// - 启用毛玻璃效果（低模糊半径）
    /// - 2x MSAA
    /// - 标准阴影质量
    Medium,
    /// 高端（高性能独立显卡）
    /// - 启用毛玻璃效果（高模糊半径）
    /// - 4x MSAA
    /// - 高质量阴影
    High,
}

impl GpuTier {
    /// 获取当前等级的描述
    pub fn description(&self) -> &'static str {
        match self {
            GpuTier::Unknown => "未检测",
            GpuTier::Low => "低端（集成显卡）",
            GpuTier::Medium => "中端（主流显卡）",
            GpuTier::High => "高端（高性能显卡）",
        }
    }

    /// 是否启用毛玻璃效果
    pub fn enable_glass_effect(&self) -> bool {
        match self {
            GpuTier::Low => false,
            GpuTier::Medium | GpuTier::High => true,
            GpuTier::Unknown => false,
        }
    }

    /// 毛玻璃模糊半径
    pub fn glass_blur_radius(&self) -> f32 {
        match self {
            GpuTier::Low => 0.0,
            GpuTier::Medium => 8.0,
            GpuTier::High => 15.0,
            GpuTier::Unknown => 0.0,
        }
    }

    /// MSAA 采样数
    pub fn msaa_samples(&self) -> u32 {
        match self {
            GpuTier::Low => 1,
            GpuTier::Medium => 2,
            GpuTier::High => 4,
            GpuTier::Unknown => 1,
        }
    }

    /// 是否启用高质量阴影
    pub fn high_quality_shadows(&self) -> bool {
        match self {
            GpuTier::Low => false,
            GpuTier::Medium | GpuTier::High => true,
            GpuTier::Unknown => false,
        }
    }

    /// 获取推荐配置
    pub fn get_recommended_config(&self) -> GpuTierConfig {
        match self {
            GpuTier::Low => GpuTierConfig {
                glass_effect: false,
                glass_blur_radius: 0.0,
                msaa_samples: 1,
                high_quality_shadows: false,
                panel_transparency: 0.95,
                animation_enabled: false,
            },
            GpuTier::Medium => GpuTierConfig {
                glass_effect: true,
                glass_blur_radius: 8.0,
                msaa_samples: 2,
                high_quality_shadows: true,
                panel_transparency: 0.85,
                animation_enabled: true,
            },
            GpuTier::High => GpuTierConfig {
                glass_effect: true,
                glass_blur_radius: 15.0,
                msaa_samples: 4,
                high_quality_shadows: true,
                panel_transparency: 0.75,
                animation_enabled: true,
            },
            GpuTier::Unknown => GpuTierConfig::default(),
        }
    }
}

impl fmt::Display for GpuTier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.description())
    }
}

/// GPU 配置（用户可手动调整）
#[derive(Debug, Clone)]
pub struct GpuTierConfig {
    /// 是否启用毛玻璃效果
    pub glass_effect: bool,
    /// 毛玻璃模糊半径
    pub glass_blur_radius: f32,
    /// MSAA 采样数
    pub msaa_samples: u32,
    /// 是否启用高质量阴影
    pub high_quality_shadows: bool,
    /// 面板透明度（0.0-1.0）
    pub panel_transparency: f32,
    /// 是否启用动画
    pub animation_enabled: bool,
}

impl Default for GpuTierConfig {
    fn default() -> Self {
        Self {
            glass_effect: false,
            glass_blur_radius: 0.0,
            msaa_samples: 1,
            high_quality_shadows: false,
            panel_transparency: 0.9,
            animation_enabled: true,
        }
    }
}

impl GpuTierConfig {
    /// 创建高端配置
    pub fn high_quality() -> Self {
        Self {
            glass_effect: true,
            glass_blur_radius: 15.0,
            msaa_samples: 4,
            high_quality_shadows: true,
            panel_transparency: 0.75,
            animation_enabled: true,
        }
    }

    /// 创建中端配置
    pub fn medium_quality() -> Self {
        Self {
            glass_effect: true,
            glass_blur_radius: 8.0,
            msaa_samples: 2,
            high_quality_shadows: true,
            panel_transparency: 0.85,
            animation_enabled: true,
        }
    }

    /// 创建低端配置
    pub fn low_quality() -> Self {
        Self {
            glass_effect: false,
            glass_blur_radius: 0.0,
            msaa_samples: 1,
            high_quality_shadows: false,
            panel_transparency: 0.95,
            animation_enabled: false,
        }
    }

    /// 应用配置到毛玻璃渲染器
    #[cfg(feature = "gpu")]
    pub fn apply_to_glass_renderer(&self, glass_renderer: &mut crate::render::GlassEffectRenderer) {
        glass_renderer.set_enabled(self.glass_effect);
        glass_renderer.set_blur_radius(self.glass_blur_radius);
    }

    /// 是否启用所有高级视觉效果
    pub fn all_effects_enabled(&self) -> bool {
        self.glass_effect && self.high_quality_shadows && self.msaa_samples > 1
    }
}

/// GPU 信息
#[derive(Debug, Clone)]
pub struct GpuInfo {
    /// GPU 名称
    pub name: String,
    /// 显存大小（MB，0 表示未知）
    pub vram_mb: u32,
    /// 是否集成显卡
    pub is_integrated: bool,
    /// 后端类型
    pub backend: wgpu::Backend,
}

impl Default for GpuInfo {
    fn default() -> Self {
        Self {
            name: String::new(),
            vram_mb: 0,
            is_integrated: false,
            backend: wgpu::Backend::Gl, // 使用 Gl 作为默认值
        }
    }
}

/// 检测 GPU 等级
#[cfg(feature = "gpu")]
pub fn detect_gpu_tier() -> (GpuTier, GpuInfo) {
    // 创建 wgpu 实例
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
    });

    // 请求适配器
    let adapter = match futures::executor::block_on(async {
        instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .await
    }) {
        Some(a) => a,
        None => return (GpuTier::Low, GpuInfo::default()),
    };

    // 获取 GPU 信息
    let info = adapter.get_info();
    let gpu_name = info.device.to_string();
    let backend = info.backend;
    let is_integrated = info.device_type == wgpu::DeviceType::IntegratedGpu;

    // 估算显存大小
    let vram_mb = match info.device_type {
        wgpu::DeviceType::DiscreteGpu => {
            // 独立显卡通常有 2GB+ 显存
            match backend {
                wgpu::Backend::Vulkan | wgpu::Backend::Dx12 | wgpu::Backend::Metal => {
                    // 尝试从适配器属性获取
                    4096
                }
                _ => 2048,
            }
        }
        wgpu::DeviceType::IntegratedGpu => {
            // 集成显卡共享系统内存
            512
        }
        _ => 1024,
    };

    // 根据 GPU 类型和后端判断等级
    let tier = match info.device_type {
        wgpu::DeviceType::DiscreteGpu => {
            // 独立显卡
            match backend {
                wgpu::Backend::Vulkan | wgpu::Backend::Dx12 | wgpu::Backend::Metal => GpuTier::High,
                wgpu::Backend::Gl => GpuTier::Medium,
                _ => GpuTier::Medium,
            }
        }
        wgpu::DeviceType::IntegratedGpu => {
            // 集成显卡
            match backend {
                wgpu::Backend::Metal => GpuTier::Medium, // Apple Silicon
                wgpu::Backend::Dx12 | wgpu::Backend::Vulkan => GpuTier::Medium,
                _ => GpuTier::Low,
            }
        }
        wgpu::DeviceType::VirtualGpu => GpuTier::Medium,
        wgpu::DeviceType::Cpu => GpuTier::Low,
        wgpu::DeviceType::Other => GpuTier::Low,
    };

    let gpu_info = GpuInfo {
        name: gpu_name,
        vram_mb,
        is_integrated,
        backend,
    };

    (tier, gpu_info)
}

/// 检测 GPU 等级（非 GPU 特性版本）
#[cfg(not(feature = "gpu"))]
pub fn detect_gpu_tier() -> (GpuTier, GpuInfo) {
    (GpuTier::Low, GpuInfo::default())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_tier_features() {
        // 低端 GPU
        assert!(!GpuTier::Low.enable_glass_effect());
        assert_eq!(GpuTier::Low.msaa_samples(), 1);
        assert!(!GpuTier::Low.high_quality_shadows());

        // 中端 GPU
        assert!(GpuTier::Medium.enable_glass_effect());
        assert_eq!(GpuTier::Medium.msaa_samples(), 2);
        assert!(GpuTier::Medium.high_quality_shadows());

        // 高端 GPU
        assert!(GpuTier::High.enable_glass_effect());
        assert_eq!(GpuTier::High.msaa_samples(), 4);
        assert!(GpuTier::High.high_quality_shadows());
    }

    #[test]
    fn test_gpu_tier_config() {
        let config = GpuTierConfig::high_quality();
        assert!(config.glass_effect);
        assert_eq!(config.msaa_samples, 4);
        assert!(config.all_effects_enabled());

        let config = GpuTierConfig::low_quality();
        assert!(!config.glass_effect);
        assert_eq!(config.msaa_samples, 1);
        assert!(!config.all_effects_enabled());
    }
}
