//! 加速器注册表实现

use accelerator_api::{Accelerator, AcceleratorOp};
use crate::strategy::SchedulingStrategy;
use crate::preferences::AcceleratorPreferences;
use log::{info, debug};

/// 加速器注册表
///
/// 负责运行时发现、注册和调度加速器后端
#[derive(Debug)]
pub struct AcceleratorRegistry {
    /// 已注册的加速器列表（按优先级排序）
    accelerators: Vec<Box<dyn Accelerator>>,
    /// 调度策略
    strategy: SchedulingStrategy,
    /// 用户偏好设置
    preferences: AcceleratorPreferences,
    /// 是否已初始化
    initialized: bool,
}

impl AcceleratorRegistry {
    /// 创建新的注册表（空）
    pub fn new() -> Self {
        Self {
            accelerators: Vec::new(),
            strategy: SchedulingStrategy::PerformanceFirst,
            preferences: AcceleratorPreferences::default(),
            initialized: false,
        }
    }

    /// 发现并注册所有可用的加速器
    ///
    /// 按优先级顺序尝试：
    /// 1. CUDA (NVIDIA GPU)
    /// 2. OpenCL (跨平台 GPU)
    /// 3. wgpu (WebGPU，支持核显/集显)
    /// 4. CPU (纯 Rust，总是可用)
    pub fn discover_all() -> Self {
        info!("开始发现可用加速器...");
        
        let mut registry = Self::new();
        
        // 注意：CUDA 和 OpenCL 后端暂未实现，预留接口
        // 未来实现时按以下顺序添加：
        
        // 1. 尝试 CUDA
        // if let Ok(cuda) = accelerator_cuda::CudaAccelerator::new() {
        //     registry.register(Box::new(cuda));
        //     info!("发现 CUDA 加速器");
        // }
        
        // 2. 尝试 OpenCL
        // if let Ok(opencl) = accelerator_opencl::OpenClAccelerator::new() {
        //     registry.register(Box::new(opencl));
        //     info!("发现 OpenCL 加速器");
        // }
        
        // 3. 尝试 wgpu（如果启用）
        #[cfg(feature = "wgpu")]
        if let Ok(wgpu) = accelerator_wgpu::WgpuAccelerator::new_sync() {
            registry.register(Box::new(wgpu));
            info!("发现 wgpu 加速器");
        }
        
        // 4. CPU fallback（总是可用）
        let cpu = accelerator_cpu::CpuAccelerator::new();
        registry.register(Box::new(cpu));
        info!("发现 CPU 加速器（fallback）");
        
        registry.initialized = true;
        info!("加速器发现完成，共发现 {} 个加速器", registry.accelerators.len());
        
        registry
    }

    /// 注册加速器
    pub fn register(&mut self, accelerator: Box<dyn Accelerator>) {
        let name = accelerator.name().to_string();
        let score = accelerator.performance_score();
        debug!("注册加速器：{} (性能评分：{:.2})", name, score);
        self.accelerators.push(accelerator);
    }

    /// 选择最佳加速器
    ///
    /// 根据调度策略和用户偏好选择
    pub fn select_best(&self, op: AcceleratorOp) -> Option<&dyn Accelerator> {
        if self.accelerators.is_empty() {
            return None;
        }

        match &self.strategy {
            SchedulingStrategy::PerformanceFirst => {
                // 选择性能最高且支持该操作的加速器
                self.accelerators
                    .iter()
                    .filter(|a| a.supports_op(op))
                    .max_by(|a, b| {
                        a.performance_score()
                            .partial_cmp(&b.performance_score())
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map(|a| a.as_ref())
            }
            SchedulingStrategy::PowerSaving => {
                // 选择功耗最低的（通常是 CPU 或集成 GPU）
                self.accelerators
                    .iter()
                    .filter(|a| a.supports_op(op))
                    .min_by(|a, b| {
                        // 简单启发：性能越低功耗越低
                        a.performance_score()
                            .partial_cmp(&b.performance_score())
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map(|a| a.as_ref())
            }
            SchedulingStrategy::Custom(criteria) => {
                // 根据自定义条件选择
                self.accelerators
                    .iter()
                    .filter(|a| a.supports_op(op))
                    .max_by(|a, b| {
                        criteria.compare(a.as_ref(), b.as_ref())
                    })
                    .map(|a| a.as_ref())
            }
        }
    }

    /// 选择最佳加速器（返回引用计数，便于共享）
    /// 
    /// 注意：此函数目前返回 CPU 加速器的 Arc
    /// TODO: 改进为持有 Arc<dyn Accelerator> 而不是 Box
    #[allow(dead_code)]
    pub fn select_best_arc(&self, _op: AcceleratorOp) -> Option<std::sync::Arc<dyn Accelerator>> {
        // 简单实现：返回 CPU fallback 的 Arc
        // 未来改进时可以返回真正的共享 Arc
        self.fallback().map(|_acc| {
            // 临时实现：总是返回 CPU
            std::sync::Arc::new(accelerator_cpu::CpuAccelerator::new()) as std::sync::Arc<dyn Accelerator>
        })
    }

    /// 获取所有已注册的加速器
    pub fn accelerators(&self) -> &[Box<dyn Accelerator>] {
        &self.accelerators
    }

    /// 获取加速器数量
    pub fn len(&self) -> usize {
        self.accelerators.len()
    }

    /// 是否为空
    pub fn is_empty(&self) -> bool {
        self.accelerators.is_empty()
    }

    /// 是否有 GPU 加速器
    pub fn has_gpu(&self) -> bool {
        self.accelerators.iter().any(|a| {
            let name = a.name().to_lowercase();
            name.contains("gpu") || name.contains("cuda") || name.contains("opencl")
        })
    }

    /// 获取调度策略
    pub fn strategy(&self) -> &SchedulingStrategy {
        &self.strategy
    }

    /// 设置调度策略
    pub fn set_strategy(&mut self, strategy: SchedulingStrategy) {
        self.strategy = strategy;
    }

    /// 获取用户偏好
    pub fn preferences(&self) -> &AcceleratorPreferences {
        &self.preferences
    }

    /// 设置用户偏好
    pub fn set_preferences(&mut self, preferences: AcceleratorPreferences) {
        self.preferences = preferences;
    }

    /// 获取默认加速器（CPU fallback）
    pub fn fallback(&self) -> Option<&dyn Accelerator> {
        self.accelerators.iter()
            .find(|a| a.name() == "CPU")
            .map(|a| a.as_ref())
    }

    /// 初始化状态
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }
}

impl Default for AcceleratorRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registry_creation() {
        let registry = AcceleratorRegistry::new();
        assert!(registry.is_empty());
        assert!(!registry.is_initialized());
    }

    #[test]
    fn test_discover_all() {
        let registry = AcceleratorRegistry::discover_all();
        assert!(!registry.is_empty());
        assert!(registry.is_initialized());
        assert!(registry.len() >= 1); // 至少有 CPU
    }

    #[test]
    fn test_has_gpu() {
        let registry = AcceleratorRegistry::discover_all();
        // CPU fallback 总是存在
        assert!(registry.fallback().is_some());
    }
}
