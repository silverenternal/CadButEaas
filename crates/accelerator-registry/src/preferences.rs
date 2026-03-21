//! 加速器偏好设置模块

use serde::{Deserialize, Serialize};

/// 加速器偏好设置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AcceleratorPreferences {
    /// 首选加速器名称（如果可用）
    pub preferred_accelerator: Option<String>,
    /// 禁用的加速器列表
    pub disabled_accelerators: Vec<String>,
    /// 是否允许回退到 CPU
    pub allow_cpu_fallback: bool,
    /// 最大内存使用率（0.0 - 1.0）
    pub max_memory_usage: f64,
    /// 是否启用异步执行
    pub enable_async: bool,
    /// 是否启用并发执行
    pub enable_concurrent: bool,
}

impl Default for AcceleratorPreferences {
    fn default() -> Self {
        Self {
            preferred_accelerator: None,
            disabled_accelerators: Vec::new(),
            allow_cpu_fallback: true,
            max_memory_usage: 0.8,
            enable_async: true,
            enable_concurrent: true,
        }
    }
}

impl AcceleratorPreferences {
    /// 创建默认偏好设置
    pub fn new() -> Self {
        Self::default()
    }

    /// 设置首选加速器
    pub fn with_preferred(mut self, name: impl Into<String>) -> Self {
        self.preferred_accelerator = Some(name.into());
        self
    }

    /// 禁用加速器
    pub fn with_disabled(mut self, names: Vec<String>) -> Self {
        self.disabled_accelerators = names;
        self
    }

    /// 设置是否允许 CPU 回退
    pub fn with_cpu_fallback(mut self, allow: bool) -> Self {
        self.allow_cpu_fallback = allow;
        self
    }

    /// 设置最大内存使用率
    pub fn with_max_memory_usage(mut self, usage: f64) -> Self {
        self.max_memory_usage = usage.clamp(0.0, 1.0);
        self
    }

    /// 检查是否禁用指定加速器
    pub fn is_disabled(&self, name: &str) -> bool {
        self.disabled_accelerators.iter().any(|n| n == name)
    }
}
