//! 测试工具模块
//!
//! 提供统一的测试文件路径处理

use std::path::PathBuf;

/// 获取项目根目录（workspace 根）
///
/// 基于 CARGO_MANIFEST_DIR 环境变量推导：
/// - 如果在 crates/parser/tests/ 下，往上三层到项目根
/// - 如果在 tests/ 下，往上一层到项目根
#[allow(dead_code)]
pub fn get_workspace_root() -> PathBuf {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR")
        .expect("CARGO_MANIFEST_DIR 应该设置");

    let path = PathBuf::from(&manifest_dir);

    // 尝试往上查找项目根（通过查找 Cargo.toml 或特定目录）
    let mut current = path.as_path();
    
    // 最多往上找 5 层
    for _ in 0..5 {
        // 检查是否有 dxfs 目录（项目根的标志）
        if current.join("dxfs").exists() {
            return current.to_path_buf();
        }
        
        if let Some(parent) = current.parent() {
            current = parent;
        } else {
            break;
        }
    }

    // 如果找不到，默认往上两层（适用于 crates/parser/tests/ 结构）
    PathBuf::from(&manifest_dir)
        .parent()
        .expect("应该有父目录")
        .parent()
        .expect("应该有祖父目录")
        .to_path_buf()
}

/// 获取 dxfs/ 目录路径
#[allow(dead_code)]
pub fn get_dxf_dir() -> PathBuf {
    get_workspace_root().join("dxfs")
}

/// 获取 testpdf/ 目录路径
#[allow(dead_code)]
pub fn get_pdf_dir() -> PathBuf {
    get_workspace_root().join("testpdf")
}

/// 列出所有 DXF 文件
#[allow(dead_code)]
pub fn list_dxf_files() -> Vec<PathBuf> {
    let dxf_dir = get_dxf_dir();
    let mut files = Vec::new();

    if let Ok(entries) = std::fs::read_dir(&dxf_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().map_or(false, |ext| ext == "dxf") {
                files.push(path);
            }
        }
    }

    files
}

/// 列出所有 PDF 文件
#[allow(dead_code)]
pub fn list_pdf_files() -> Vec<PathBuf> {
    let pdf_dir = get_pdf_dir();
    let mut files = Vec::new();

    if let Ok(entries) = std::fs::read_dir(&pdf_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().map_or(false, |ext| ext == "pdf") {
                files.push(path);
            }
        }
    }

    files
}

/// 查找第一个可解析的 DXF 文件
///
/// 用于测试中避免硬编码文件名，支持动态适配测试数据
#[allow(dead_code)]
pub fn find_first_parseable_dxf(
    dxf_dir: &PathBuf,
    parser: &parser::DxfParser,
) -> Option<PathBuf> {
    if !dxf_dir.exists() {
        return None;
    }

    let mut files: Vec<PathBuf> = list_dxf_files();
    files.sort(); // 排序确保稳定性

    for file_path in &files {
        if parser.parse_file(file_path).is_ok() {
            return Some(file_path.clone());
        }
    }

    // 如果都解析失败，返回第一个 DXF 文件（让测试失败并暴露问题）
    files.into_iter().next()
}
