/// 集成测试工具模块
///
/// 提供获取工作目录、DXF 测试文件目录、PDF 测试文件目录的工具函数。

use std::path::PathBuf;

/// 获取 Cargo 工作空间根目录
///
/// 基于 `CARGO_MANIFEST_DIR` 环境变量向上查找至包含 `Cargo.toml` 的目录。
///
/// # 返回
/// - `PathBuf`: 工作空间根目录路径
///
/// # Panics
/// - 如果 `CARGO_MANIFEST_DIR` 未设置
/// - 如果工作空间根目录不存在
pub fn get_workspace_root() -> PathBuf {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR")
        .expect("CARGO_MANIFEST_DIR 应该设置");
    
    let manifest_path = PathBuf::from(manifest_dir);
    
    // 向上查找至包含 Cargo.toml 的目录（工作空间根目录）
    let mut current = manifest_path.as_path();
    while let Some(parent) = current.parent() {
        if parent.join("Cargo.toml").exists() {
            return parent.to_path_buf();
        }
        current = parent;
    }
    
    // 如果没找到，返回 manifest_dir 的父目录
    manifest_path.parent().unwrap().to_path_buf()
}

/// 获取 DXF 测试文件目录
///
/// 返回工作空间根目录下的 `dxfs/` 目录。
///
/// # 返回
/// - `PathBuf`: DXF 测试文件目录路径
pub fn get_dxf_dir() -> PathBuf {
    get_workspace_root().join("dxfs")
}

/// 获取 PDF 测试文件目录
///
/// 返回工作空间根目录下的 `testpdf/` 目录。
///
/// # 返回
/// - `PathBuf`: PDF 测试文件目录路径
pub fn get_pdf_dir() -> PathBuf {
    get_workspace_root().join("testpdf")
}
