//! OpenCV 配置模块
//!
//! 提供 OpenCV 运行时配置和初始化工具

use opencv::core::{get_num_threads, set_num_threads};
use tracing::{info, warn};

/// OpenCV 线程配置
#[derive(Debug, Clone)]
pub struct OpenCvThreadConfig {
    /// 线程数设置
    /// - `None`: 使用 OpenCV 默认设置
    /// - `Some(0)`: 使用所有可用 CPU 核心
    /// - `Some(n)`: 使用 n 个线程
    pub threads: Option<i32>,
}

impl Default for OpenCvThreadConfig {
    fn default() -> Self {
        Self { threads: Some(0) } // 默认使用所有核心
    }
}

/// 初始化 OpenCV 运行时配置
///
/// # 参数
/// * `config` - 线程配置
///
/// # 返回
/// 成功时返回实际设置的线程数，失败时返回错误信息
pub fn init_opencv_runtime(config: &OpenCvThreadConfig) -> Result<i32, String> {
    if let Some(threads) = config.threads {
        set_num_threads(threads).map_err(|e| format!("设置 OpenCV 线程数失败：{}", e))?;

        let actual_threads =
            get_num_threads().map_err(|e| format!("获取 OpenCV 线程数失败：{}", e))?;

        info!(
            target: "vectorize::opencv",
            threads = actual_threads,
            "OpenCV 运行时初始化完成"
        );

        Ok(actual_threads)
    } else {
        let actual_threads =
            get_num_threads().map_err(|e| format!("获取 OpenCV 线程数失败：{}", e))?;

        info!(
            target: "vectorize::opencv",
            threads = actual_threads,
            "使用 OpenCV 默认线程配置"
        );

        Ok(actual_threads)
    }
}

/// 获取 OpenCV 版本信息
pub fn get_opencv_version() -> String {
    opencv::core::get_version_string().unwrap_or_else(|_| "unknown".to_string())
}

/// 检查 OpenCV 版本兼容性
pub fn check_opencv_version() -> Vec<String> {
    let mut warnings = Vec::new();

    match opencv::core::get_version_string() {
        Ok(version) => {
            // 检查已知有问题的版本
            if version.starts_with("4.5.0") {
                warnings.push(
                    "OpenCV 4.5.0 有已知的 Canny 边缘检测 bug，建议升级到 4.5.1+".to_string(),
                );
            }
            if version.starts_with("4.6.0") {
                warnings
                    .push("OpenCV 4.6.0 在某些系统上可能有稳定性问题，建议升级到 4.8+".to_string());
            }
        }
        Err(_) => {
            warnings.push("无法获取 OpenCV 版本".to_string());
        }
    }

    if !warnings.is_empty() {
        for warning in &warnings {
            warn!(target: "vectorize::opencv", "{}", warning);
        }
    }

    warnings
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_opencv_version() {
        let version = get_opencv_version();
        assert!(!version.is_empty());
        assert_ne!(version, "unknown");
    }

    #[test]
    fn test_init_opencv_runtime() {
        let config = OpenCvThreadConfig::default();
        let result = init_opencv_runtime(&config);
        assert!(result.is_ok());
        assert!(result.unwrap() > 0);
    }
}
