//! DWG 文件解析器
//!
//! DWG 是 AutoCAD 的专有二进制格式。由于 Rust 生态没有成熟的 DWG 解析库，
//! 本解析器通过外部转换工具将 DWG 转为 DXF，然后委托给现有 DXF 解析器处理。
//!
//! ## 支持的转换工具（自动检测）
//!
//! 1. **dwg2dxf** (libredwg) — 推荐，开源，支持 R13-R2018
//! 2. **ODAFileConverter** (Open Design Alliance) — 商业工具，支持所有版本
//! 3. **TeighaFileConverter** — ODA 旧名称
//!
//! ## 安装 libredwg
//!
//! ```bash
//! # Ubuntu/Debian
//! sudo apt install libredwg-tools
//!
//! # Arch Linux
//! sudo pacman -S libredwg
//!
//! # macOS
//! brew install libredwg
//!
//! # 从源码编译
//! git clone https://github.com/LibreDWG/libredwg.git
//! cd libredwg && ./autogen.sh && ./configure && make && sudo make install
//! ```

use crate::{DxfParseReport, DxfParserEnum};
use common_types::{CadError, InternalErrorReason, RawEntity};
use std::path::{Path, PathBuf};
use std::process::Command;

/// DWG 转换工具类型
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DwgConverter {
    /// dwg2dxf (libredwg)
    LibreDWG,
    /// ODA File Converter
    ODA,
    /// 未找到转换工具
    NotFound,
}

impl DwgConverter {
    /// 检测系统上可用的 DWG 转换工具
    pub fn detect() -> Self {
        if Self::command_exists("dwg2dxf") {
            Self::LibreDWG
        } else if Self::command_exists("ODAFileConverter")
            || Self::command_exists("TeighaFileConverter")
        {
            Self::ODA
        } else {
            Self::NotFound
        }
    }

    /// 将 DWG 文件转换为临时 DXF 文件
    pub fn convert_to_dxf(&self, dwg_path: &Path) -> Result<PathBuf, CadError> {
        let temp_dxf = Self::temp_dxf_path(dwg_path)?;

        match self {
            DwgConverter::LibreDWG => {
                let output = Command::new("dwg2dxf")
                    .arg("-o")
                    .arg(&temp_dxf)
                    .arg(dwg_path)
                    .output()
                    .map_err(|e| {
                        CadError::internal(InternalErrorReason::Panic {
                            message: format!("dwg2dxf 执行失败: {}", e),
                        })
                    })?;

                if !output.status.success() {
                    let stderr = String::from_utf8_lossy(&output.stderr);
                    return Err(CadError::UnsupportedFormat {
                        format: "dwg".to_string(),
                        supported_formats: vec![format!(
                            "dwg 转换失败 (dwg2dxf): {}",
                            stderr.lines().next().unwrap_or("未知错误")
                        )],
                    });
                }
            }
            DwgConverter::ODA => {
                // ODA File Converter 需要: input_dir output_dir version acad_type
                let input_dir = dwg_path.parent().unwrap_or(Path::new("."));
                let output_dir = temp_dxf.parent().unwrap_or(Path::new("."));
                let file_name = dwg_path.file_name().unwrap_or_default();

                let output = Command::new("ODAFileConverter")
                    .arg(input_dir)
                    .arg(output_dir)
                    .arg("ACAD2018") // 目标版本
                    .arg("DWG") // acad_type
                    .arg(file_name)
                    .output()
                    .map_err(|e| {
                        CadError::internal(InternalErrorReason::Panic {
                            message: format!("ODAFileConverter 执行失败: {}", e),
                        })
                    })?;

                if !output.status.success() {
                    let stderr = String::from_utf8_lossy(&output.stderr);
                    return Err(CadError::UnsupportedFormat {
                        format: "dwg".to_string(),
                        supported_formats: vec![format!(
                            "dwg 转换失败 (ODA): {}",
                            stderr.lines().next().unwrap_or("未知错误")
                        )],
                    });
                }
            }
            DwgConverter::NotFound => {
                return Err(CadError::UnsupportedFormat {
                    format: "dwg".to_string(),
                    supported_formats: vec![
                        "未安装 DWG 转换工具。请安装 libredwg (dwg2dxf) 或 ODA File Converter".to_string(),
                        "安装 libredwg: sudo apt install libredwg-tools (Ubuntu) 或 brew install libredwg (macOS)".to_string(),
                    ],
                });
            }
        }

        if !temp_dxf.exists() {
            return Err(CadError::UnsupportedFormat {
                format: "dwg".to_string(),
                supported_formats: vec!["DWG 转换完成但未生成 DXF 文件".to_string()],
            });
        }

        Ok(temp_dxf)
    }

    fn command_exists(cmd: &str) -> bool {
        Command::new(cmd).arg("--version").output().is_ok()
            || Command::new(cmd).arg("-h").output().is_ok()
    }

    fn temp_dxf_path(dwg_path: &Path) -> Result<PathBuf, CadError> {
        let stem = dwg_path
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "converted".to_string());
        let temp_dir = std::env::temp_dir();
        let mut path = temp_dir.join(format!("cad_{}_{}.dxf", stem, std::process::id()));

        // 如果文件已存在（进程 ID 重用），添加时间戳
        if path.exists() {
            let ts = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_micros())
                .unwrap_or(0);
            path = temp_dir.join(format!("cad_{}_{}_{}.dxf", stem, std::process::id(), ts));
        }

        Ok(path)
    }
}

/// DWG 解析器
///
/// 通过外部转换工具将 DWG 转为 DXF，然后委托给 DXF 解析器处理
pub struct DwgParser {
    converter: DwgConverter,
    dxf_parser: DxfParserEnum,
}

impl DwgParser {
    /// 创建新的 DWG 解析器
    pub fn new() -> Self {
        Self {
            converter: DwgConverter::detect(),
            dxf_parser: crate::ParserFactory::create_default().expect("DXF parser creation"),
        }
    }

    /// 使用自定义 DXF 解析器创建
    pub fn with_dxf_parser(dxf_parser: DxfParserEnum) -> Self {
        Self {
            converter: DwgConverter::detect(),
            dxf_parser,
        }
    }

    /// 解析 DWG 文件
    pub fn parse_file(&self, path: impl AsRef<Path>) -> Result<Vec<RawEntity>, CadError> {
        let (entities, _report) = self.parse_file_with_report(path)?;
        Ok(entities)
    }

    /// 解析 DWG 文件并返回报告
    pub fn parse_file_with_report(
        &self,
        path: impl AsRef<Path>,
    ) -> Result<(Vec<RawEntity>, DxfParseReport), CadError> {
        let path = path.as_ref();

        // 转换 DWG → DXF
        let temp_dxf = self.converter.convert_to_dxf(path)?;

        // 委托给 DXF 解析器
        let result = self.dxf_parser.parse_file_with_report(&temp_dxf);

        // 清理临时文件
        let _ = std::fs::remove_file(&temp_dxf);

        result
    }

    /// 获取使用的转换工具
    pub fn converter(&self) -> DwgConverter {
        self.converter
    }
}

impl Default for DwgParser {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_converter_detect() {
        // 检测应该在无异常的情况下完成
        let converter = DwgConverter::detect();
        // 大多数开发环境没有安装转换工具，所以预期 NotFound
        // 测试仅确保检测逻辑不崩溃
        let _ = format!("{:?}", converter);
    }

    #[test]
    fn test_converter_not_found_error() {
        let converter = DwgConverter::NotFound;
        let result = converter.convert_to_dxf(Path::new("test.dwg"));
        assert!(result.is_err());

        if let Err(CadError::UnsupportedFormat { format, .. }) = result {
            assert_eq!(format, "dwg");
        } else {
            panic!("Expected UnsupportedFormat error");
        }
    }

    #[test]
    fn test_dwg_parser_creation() {
        let parser = DwgParser::new();
        // 创建不应崩溃
        let _ = parser.converter();
    }

    #[test]
    fn test_dwg_parser_with_dxf_parser() {
        let dxf_parser = crate::ParserFactory::create_default().expect("parser");
        let parser = DwgParser::with_dxf_parser(dxf_parser);
        let _ = parser.converter();
    }
}
