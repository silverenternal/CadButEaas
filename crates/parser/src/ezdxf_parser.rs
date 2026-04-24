//! ezdxf 解析器桥接模块
//!
//! 通过 subprocess 调用 Python ezdxf 库解析 DXF 文件，
//! 输出与 Rust `RawEntity` 兼容的 JSON。
//!
//! 解析失败时自动降级到 Rust `dxf` crate 解析器。

use common_types::{CadError, DxfParseReason, InternalErrorReason, IoErrorReason, RawEntity};
use serde::Deserialize;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};

use crate::dxf_parser::DxfParser;
use crate::parser_trait::DxfParserTrait;
use crate::{DxfConfig, DxfParseReport};

/// 用于生成唯一临时文件名的计数器
static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

/// ezdxf 解析器
///
/// 主路径：Python ezdxf 解析
/// 降级路径：Rust dxf crate 解析
#[derive(Clone)]
pub struct EzdxfParser {
    /// 降级用的 Rust 解析器
    fallback: DxfParser,
    /// Python 解释器路径（可选，默认使用项目 .venv）
    python_path: Option<PathBuf>,
    /// Python 解析脚本路径
    script_path: PathBuf,
    /// 是否强制使用 fallback（当 Python 不可用时自动设置）
    force_fallback: bool,
}

impl EzdxfParser {
    /// 创建新的 ezdxf 解析器
    pub fn new() -> Self {
        let script_path = find_script_path();
        let python_path = find_python_path();
        let force_fallback = !check_python_available(&python_path, &script_path);

        Self {
            fallback: DxfParser::new(),
            python_path,
            script_path,
            force_fallback,
        }
    }

    /// 设置降级解析器的图层过滤器
    pub fn with_layer_filter(mut self, layers: Vec<String>) -> Self {
        self.fallback = self.fallback.with_layer_filter(layers);
        self
    }

    /// 设置 Python 解释器路径
    pub fn with_python_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.python_path = Some(path.into());
        self.force_fallback = !check_python_available(&self.python_path, &self.script_path);
        self
    }

    /// 是否正在使用 fallback
    pub fn is_using_fallback(&self) -> bool {
        self.force_fallback
    }

    /// 使用 ezdxf 解析 DXF 文件，失败时自动降级
    pub fn parse_file(&self, path: impl AsRef<Path>) -> Result<Vec<RawEntity>, CadError> {
        let path = path.as_ref();

        if self.force_fallback {
            tracing::debug!("ezdxf: using fallback Rust dxf parser");
            return self.fallback.parse_file(path);
        }

        let python_path = self
            .python_path
            .as_deref()
            .unwrap_or_else(|| Path::new("python3"));

        let result = run_python_parser(python_path, &self.script_path, path);

        match result {
            Ok(entities) => Ok(entities),
            Err(e) => {
                tracing::warn!("ezdxf parse failed, falling back to Rust dxf parser: {}", e);
                self.fallback.parse_file(path)
            }
        }
    }

    /// 使用 ezdxf 解析字节流，失败时自动降级
    pub fn parse_bytes(&self, bytes: &[u8]) -> Result<Vec<RawEntity>, CadError> {
        // Python subprocess 需要文件路径，先写入临时文件
        // 使用 PID + 原子计数器确保多线程安全
        let counter = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
        let temp_path = std::env::temp_dir().join(format!(
            "cadbut_eaas_ezdxf_{}_{}.dxf",
            std::process::id(),
            counter
        ));

        std::fs::write(&temp_path, bytes)
            .map_err(|e| CadError::io_path(&temp_path, IoErrorReason::ReadFailed, e))?;

        let result = self.parse_file(&temp_path);

        // 清理临时文件
        let _ = std::fs::remove_file(&temp_path);

        result
    }
}

impl Default for EzdxfParser {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// DxfParserTrait 实现
// ============================================================================

#[async_trait::async_trait]
impl DxfParserTrait for EzdxfParser {
    fn parse_file_with_report(
        &self,
        path: impl AsRef<Path>,
    ) -> Result<(Vec<RawEntity>, DxfParseReport), CadError> {
        let entities = self.parse_file(path)?;
        let report = build_report_from_entities(&entities);
        Ok((entities, report))
    }

    async fn parse_file_async(
        &self,
        path: impl AsRef<Path> + Send,
    ) -> Result<Vec<RawEntity>, CadError> {
        let path = path.as_ref().to_path_buf();
        let parser = self.clone();

        tokio::task::spawn_blocking(move || parser.parse_file(path))
            .await
            .map_err(|e| {
                CadError::internal(InternalErrorReason::Panic {
                    message: format!("tokio join error: {}", e),
                })
            })?
    }

    fn parse_bytes_with_report(
        &self,
        bytes: &[u8],
    ) -> Result<(Vec<RawEntity>, DxfParseReport), CadError> {
        let entities = self.parse_bytes(bytes)?;
        let report = build_report_from_entities(&entities);
        Ok((entities, report))
    }

    fn config(&self) -> &DxfConfig {
        &self.fallback.config
    }

    fn name(&self) -> &'static str {
        "EzdxfParser"
    }
}

/// 从解析结果构建简单的统计报告
fn build_report_from_entities(entities: &[RawEntity]) -> DxfParseReport {
    let mut layer_distribution = std::collections::HashMap::new();
    let mut entity_type_distribution = std::collections::HashMap::new();

    for entity in entities {
        // 统计图层
        if let Some(layer) = entity.layer() {
            *layer_distribution.entry(layer.to_string()).or_insert(0) += 1;
        }

        // 统计实体类型
        let type_name = entity.entity_type_name();
        *entity_type_distribution
            .entry(type_name.to_string())
            .or_insert(0) += 1;
    }

    DxfParseReport {
        layer_distribution,
        entity_type_distribution,
        ..DxfParseReport::default()
    }
}

// ============================================================================
// Python 发现与检测
// ============================================================================

/// 查找 Python 解析脚本路径
fn find_script_path() -> PathBuf {
    // 尝试多个可能的路径
    let candidates = [
        // 从 crates/parser/ 根目录
        PathBuf::from("python/dxf_parser.py"),
        // 从 workspace 根目录
        PathBuf::from("crates/parser/python/dxf_parser.py"),
        // 从可执行文件所在目录
        {
            let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
            p.push("python/dxf_parser.py");
            p
        },
    ];

    for candidate in &candidates {
        if candidate.is_file() {
            return candidate.clone();
        }
    }

    // 默认返回相对于 CARGO_MANIFEST_DIR 的路径
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("python/dxf_parser.py");
    p
}

/// 查找 Python 解释器路径
fn find_python_path() -> Option<PathBuf> {
    // 尝试项目 .venv
    let workspace_venv = find_workspace_root().join(".venv/bin/python3");
    if workspace_venv.is_file() {
        return Some(workspace_venv);
    }

    // 尝试系统 python3
    if Path::new("python3").exists() || which_python3().is_some() {
        return None; // None = 使用 "python3" 命令
    }

    None
}

/// 查找工作区根目录（包含 Cargo.toml 且 [workspace] 的目录）
fn find_workspace_root() -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let mut current = manifest.as_path();

    loop {
        let cargo_toml = current.join("Cargo.toml");
        if cargo_toml.is_file() {
            if let Ok(content) = std::fs::read_to_string(&cargo_toml) {
                if content.contains("[workspace]") {
                    return current.to_path_buf();
                }
            }
        }

        match current.parent() {
            Some(parent) => current = parent,
            None => break,
        }
    }

    // 回退到 manifest 的祖父目录（crates/parser/ -> workspace root）
    manifest
        .parent()
        .and_then(|p| p.parent())
        .unwrap_or(manifest.as_path())
        .to_path_buf()
}

fn which_python3() -> Option<PathBuf> {
    // 简单的 PATH 查找
    if let Ok(output) = Command::new("which").arg("python3").output() {
        if output.status.success() {
            let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if !path.is_empty() {
                return Some(PathBuf::from(path));
            }
        }
    }
    None
}

/// 检测 Python 解析器是否可用
fn check_python_available(python_path: &Option<PathBuf>, script_path: &PathBuf) -> bool {
    let python_path = python_path
        .as_ref()
        .map(|p| p.as_path())
        .unwrap_or_else(|| Path::new("python3"));

    if !script_path.is_file() {
        tracing::debug!("ezdxf: script not found at {:?}", script_path);
        return false;
    }

    let result = Command::new(python_path)
        .arg("-c")
        .arg("import ezdxf; print(ezdxf.__version__)")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output();

    match result {
        Ok(output) if output.status.success() => true,
        Ok(output) => {
            let stderr = String::from_utf8_lossy(&output.stderr);
            tracing::debug!("ezdxf: python check failed: {}", stderr.trim());
            false
        }
        Err(e) => {
            tracing::debug!("ezdxf: python not available: {}", e);
            false
        }
    }
}

// ============================================================================
// Python subprocess 调用
// ============================================================================

/// 运行 Python ezdxf 解析器
fn run_python_parser(
    python_path: &Path,
    script_path: &Path,
    dxf_path: &Path,
) -> Result<Vec<RawEntity>, CadError> {
    let output = Command::new(python_path)
        .arg(script_path)
        .arg(dxf_path)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .map_err(|e| {
            CadError::internal(InternalErrorReason::Panic {
                message: format!("Failed to spawn python process: {}", e),
            })
        })?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(CadError::dxf_parse(
            dxf_path,
            DxfParseReason::ParseError(format!(
                "Python parser exited with error: {}",
                stderr.trim()
            )),
        ));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let parse_result: EzdxfResult = serde_json::from_str(&stdout).map_err(|e| {
        CadError::dxf_parse(
            dxf_path,
            DxfParseReason::ParseError(format!("Failed to parse JSON output: {}", e)),
        )
    })?;

    if !parse_result.success {
        let errors: Vec<String> = parse_result
            .errors
            .into_iter()
            .map(|e| format!("{:?}", e))
            .collect();
        return Err(CadError::dxf_parse(
            dxf_path,
            DxfParseReason::ParseError(format!("ezdxf parse failed: {}", errors.join("; "))),
        ));
    }

    Ok(parse_result.entities)
}

// ============================================================================
// ezdxf JSON 输出反序列化
// ============================================================================

#[derive(Debug, Deserialize)]
struct EzdxfResult {
    success: bool,
    entities: Vec<RawEntity>,
    errors: Vec<serde_json::Value>,
}

// ============================================================================
// RawEntity 反序列化
//
// 注意：Python 输出的 JSON 必须与 Rust RawEntity 的 serde 格式完全一致。
// RawEntity 使用 #[serde(tag = "type", rename_all = "snake_case")]
// 因此 JSON 中的 "type" 字段值应为 "line", "polyline", "arc" 等。
// ============================================================================

// RawEntity 及其相关类型已经在 common_types 中实现了 Deserialize，
// 所以我们可以直接反序列化。

// ============================================================================
// 测试
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ezdxf_parser_new() {
        let parser = EzdxfParser::new();
        // 无论 Python 是否可用，构造都应该成功
        assert!(!parser.script_path.as_os_str().is_empty());
    }

    #[test]
    fn test_find_script_path() {
        let path = find_script_path();
        // 脚本路径应该非空
        assert!(!path.as_os_str().is_empty());
    }

    #[test]
    fn test_ezdxf_parser_with_layer_filter() {
        let parser = EzdxfParser::new().with_layer_filter(vec!["Wall".to_string()]);
        assert!(parser.fallback.layer_filter.is_some());
        assert_eq!(parser.fallback.layer_filter.unwrap(), vec!["Wall"]);
    }
}
